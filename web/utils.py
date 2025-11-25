# utils.py (最終修正版 - FFmpeg 系統轉檔)
import os
import time
import json
import shutil
import subprocess  # <--- 新增：用於呼叫系統 FFmpeg
from datetime import datetime
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torchaudio
import config


# ===== 核心音訊載入函數 (FFmpeg 轉檔 + Torchaudio 讀取) =====

def load_wav_to_numpy(path: str, sample_rate: int) -> tuple[np.ndarray | None, str | None]:
    """
    1. 使用系統 FFmpeg 指令將 WebM 轉為 WAV。
    2. 使用 Torchaudio 讀取 WAV。
    3. 執行音量正規化。
    """
    try:
        audio_path = Path(path)
        print(f"🔍 [Audio] 收到檔案: {audio_path.name} (Size: {audio_path.stat().st_size} bytes)")

        # 定義轉換後的臨時 WAV 路徑
        wav_path = audio_path.with_suffix('.wav')

        # 1. 使用 subprocess 呼叫系統 FFmpeg 進行強制轉檔
        # 指令: ffmpeg -i input.webm -ar 16000 -ac 1 output.wav -y
        print(f"   🛠️ 正在呼叫 FFmpeg 將 WebM 轉為 WAV...")

        command = [
            "ffmpeg",
            "-i", str(audio_path),  # 輸入檔案
            "-ar", str(sample_rate),  # 設定採樣率 (16000)
            "-ac", "1",  # 設定單聲道
            "-y",  # 強制覆蓋
            str(wav_path)  # 輸出檔案
        ]

        # 執行指令 (隱藏視窗)
        if os.name == 'nt':  # Windows 下隱藏黑窗
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            result = subprocess.run(command, capture_output=True, text=True, startupinfo=startupinfo)
        else:
            result = subprocess.run(command, capture_output=True, text=True)

        # 檢查轉檔是否成功
        if result.returncode != 0:
            print(f"❌ FFmpeg 轉檔失敗: {result.stderr}")
            return None, None

        print(f"   ✅ 轉檔成功: {wav_path.name}")

        # 2. 使用 Torchaudio 讀取標準 WAV (這絕對不會錯)
        waveform, sr = torchaudio.load(str(wav_path))

        # 3. (保險) 再次確認採樣率
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
            waveform = resampler(waveform)

        # 4. 轉單聲道 (如果 FFmpeg 沒轉好的話)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 5. 🚨 強制音量正規化 (Normalize)
        max_val = torch.abs(waveform).max()
        if max_val > 0:
            scale_factor = 0.9 / max_val
            waveform = waveform * scale_factor
            print(f"   🔊 音量放大倍率: {scale_factor:.2f}")
        else:
            print("   ⚠️ 警告：讀取到的音訊是完全靜音！")

        # 6. 儲存除錯檔案 (這是真正餵給 Whisper 的聲音)
        debug_path = config.PROJECT_DIR / "debug_whisper_input.wav"
        torchaudio.save(str(debug_path), waveform, sample_rate)

        # 7. 轉換為 Numpy
        audio_np = waveform.squeeze().numpy()

        return audio_np, str(wav_path)

    except FileNotFoundError:
        print("❌ 錯誤：找不到 FFmpeg，請確認已安裝並加入環境變數 PATH 中！")
        return None, None
    except Exception as e:
        print(f"❌ load_wav_to_numpy 失敗: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# ===== 日誌系統 =====

class ConversationLogger:
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_log = []

    def log_turn(self, turn_data: dict):
        turn_data["timestamp"] = datetime.now().isoformat()
        turn_data["session_id"] = self.session_id
        self.session_log.append(turn_data)
        if config.ENABLE_CONVERSATION_LOG:
            log_file = self.log_dir / f"conversation_{self.session_id}.jsonl"
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(turn_data, ensure_ascii=False) + "\n")

    def get_session_summary(self) -> dict:
        if not self.session_log: return {}
        emotions = [t.get("final_emotion") for t in self.session_log if t.get("final_emotion")]
        personalities = [t.get("personality") for t in self.session_log if t.get("personality")]
        return {
            "total_turns": len(self.session_log),
            "emotions": dict(zip(*np.unique(emotions, return_counts=True))) if emotions else {},
            "personalities": dict(zip(*np.unique(personalities, return_counts=True))) if personalities else {},
            "duration": (datetime.now() - datetime.fromisoformat(self.session_log[0]["timestamp"])).total_seconds()
        }


def debug_log(message: str, level: str = "INFO"):
    if config.ENABLE_DEBUG_LOG:
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"   [{timestamp}] {level} {message}")


# utils.py (請替換 class EmotionMemory)

class EmotionMemory:
    def __init__(self, max_history=10):
        self.history = deque(maxlen=max_history)

    def add_emotion(self, text_emotion: str, audio_emotion: str, facial_emotion: str = None):
        """記錄每一輪的三種情緒"""
        self.history.append({
            "text_emotion": text_emotion,
            "audio_emotion": audio_emotion,
            "facial_emotion": facial_emotion,
            "timestamp": time.time()
        })

    def get_dominant_emotion(self) -> str:
        """取得當前最具優勢的情緒"""
        if not self.history: return "neutral"
        emotions = []
        # 只看最近一輪
        record = self.history[-1]
        if record.get("facial_emotion"): emotions.append(record["facial_emotion"])
        if record.get("audio_emotion"): emotions.append(record["audio_emotion"])
        if record.get("text_emotion"): emotions.append(record["text_emotion"])

        if not emotions: return "neutral"
        # 找出出現最多次的情緒，若平手則取第一個
        return max(set(emotions), key=emotions.count)

    def get_emotion_trend(self) -> str:
        """
        計算情緒趨勢：
        - worsening: 變差
        - improving: 變好
        - persistent_worsening: 持續低潮 (一直在 -1)
        - persistent_improving: 持續正向 (一直在 +1)
        - stable: 穩定 (一直在 0)
        """
        if len(self.history) < 2:
            return "stable"

        # 1. 定義情緒權重
        negative_emotions = {"sad", "sadness", "angry", "anger", "fear", "disgust", "frustrated", "negative", "pain",
                             "worried"}
        positive_emotions = {"happy", "happiness", "joy", "positive", "excitement", "cheerful", "encouraging",
                             "surprised", "surprise"}

        scores = []
        for record in self.history:
            turn_emotions = [
                record.get("text_emotion"),
                record.get("audio_emotion"),
                record.get("facial_emotion")
            ]

            turn_score = 0
            # 負面優先計分
            if any(str(e).lower() in negative_emotions for e in turn_emotions if e):
                turn_score = -1
            elif any(str(e).lower() in positive_emotions for e in turn_emotions if e):
                turn_score = 1

            scores.append(turn_score)

        # 2. 比較趨勢
        mid = len(scores) // 2
        if mid == 0: return "stable"

        older_scores = scores[:mid]
        recent_scores = scores[mid:]

        older_avg = sum(older_scores) / len(older_scores)
        recent_avg = sum(recent_scores) / len(recent_scores)

        diff = recent_avg - older_avg

        # 3. 判斷邏輯 (新增持續性判斷)

        # A. 有明顯變化的情況
        if diff < -0.1:  # 變差
            return "worsening"
        elif diff > 0.1:  # 變好
            return "improving"

        # B. 沒變化 (Stable)，但要看是「哪種」Stable
        else:
            if recent_avg <= -0.5:  # 分數持續為負
                return "persistent_worsening"
            elif recent_avg >= 0.5:  # 分數持續為正
                return "persistent_improving"
            else:
                return "stable"  # 真正的平淡 (0)

# 空函數，避免 server.py 報錯
def initialize_audio_settings(): pass


def get_audio_device_index(): return None


def play_audio(path): pass


def _audio_callback(*args): pass


def record_until_silence(*args): return None


def face_capture_worker(*args): pass


def clean_image_dir(): pass


def download_model(m, u, j, ju): pass


def has_wake_word(t): return config.WAKE_WORD in t.replace(" ", "")


def detect_language(text: str) -> str:
    return "zh" if any('\u4e00' <= c <= '\u9fff' for c in text) else "en"


def check_personality_switch_command(text: str, language: str) -> tuple:
    for pid, cfg in config.PERSONALITY_CONFIGS.items():
        if any(cmd in text.lower() for cmd in
               (cfg["switch_commands_zh"] if language == "zh" else cfg["switch_commands_en"])): return (True, pid)
    return (False, None)


def format_emotion_display(t, a, f, l): return ""