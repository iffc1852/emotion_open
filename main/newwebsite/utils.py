# utils.py (情緒分流版 - 解決情緒誤判)
import os
import time
import json
import shutil
import subprocess
from datetime import datetime
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torchaudio
import noisereduce as nr
import config


# ===== 核心音訊載入函數 (分流處理) =====

def load_wav_to_numpy(path: str, sample_rate: int) -> tuple[np.ndarray | None, str | None]:
    """
    1. FFmpeg 轉檔 -> 產生 raw_wav (給情緒模型用，保留原始語氣和音量)
    2. 降噪 + 正規化 -> 產生 audio_np (給 Whisper 用，追求清晰度)
    """
    try:
        audio_path = Path(path)
        print(f"🔍 [Audio] 收到檔案: {audio_path.name} (Size: {audio_path.stat().st_size} bytes)")

        # 1. 定義原始 WAV 路徑 (給情緒模型)
        raw_wav_path = audio_path.with_suffix('.wav')

        # 2. FFmpeg 轉檔 (WebM -> WAV, 16kHz, Mono)
        # 這是最原始的聲音，沒有經過 Python 的任何數位放大
        command = [
            "ffmpeg",
            "-i", str(audio_path),
            "-ar", str(sample_rate),
            "-ac", "1",
            "-y",
            str(raw_wav_path)
        ]

        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            subprocess.run(command, capture_output=True, text=True, startupinfo=startupinfo)
        else:
            subprocess.run(command, capture_output=True, text=True)

        # 3. 讀取檔案準備給 Whisper 處理
        waveform, sr = torchaudio.load(str(raw_wav_path))

        # 重取樣與轉單聲道 (確保格式對)
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 4. [Whisper 專用] 降噪處理
        audio_np_raw = waveform.squeeze().numpy()
        cleaned_audio = audio_np_raw  # 預設不變

        if len(audio_np_raw) > sample_rate * 0.1:
            try:
                # 只對 Whisper 的輸入做降噪，情緒模型聽原始的
                cleaned_audio = nr.reduce_noise(y=audio_np_raw, sr=sample_rate, stationary=True, prop_decrease=0.90)
                # print(f"   ✨ Whisper 專用降噪完成")
            except Exception:
                pass

        # 5. [Whisper 專用] 音量正規化 (放大)
        waveform_stt = torch.from_numpy(cleaned_audio).unsqueeze(0)
        max_val = torch.abs(waveform_stt).max()
        if max_val > 0:
            scale_factor = 0.9 / max_val
            waveform_stt = waveform_stt * scale_factor
            print(f"   🔊 STT 音量放大: {scale_factor:.2f} (原始路徑保留原音量)")

        # 6. 轉換為 Numpy (給 Whisper)
        audio_np_final = waveform_stt.squeeze().numpy()

        # 7. 回傳
        # 參數 1: audio_np_final (大聲、乾淨 -> 給 Whisper)
        # 參數 2: str(raw_wav_path) (原始音量、含背景音 -> 給 Emotion2Vec)
        # 讓情緒模型聽到真實的「環境音」和「微弱語氣」，而不是被暴力放大的聲音
        return audio_np_final, str(raw_wav_path)

    except Exception as e:
        print(f"❌ load_wav_to_numpy 失敗: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# ===== 日誌與記憶系統 (保持不變) =====
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

    def get_session_summary(self) -> dict: return {}


def debug_log(message: str, level: str = "INFO"):
    if config.ENABLE_DEBUG_LOG:
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"   [{timestamp}] {level} {message}")



##情緒趨勢
class EmotionMemory:
    def __init__(self, max_history=10):
        self.history = deque(maxlen=max_history)
        self.worsening_count = 0  # <--- [新增] 追蹤連續負面情緒回合數

    def add_emotion(self, text_emotion: str, audio_emotion: str, facial_emotion: str = None, winner: str = None):
        # 1. 判斷本輪情緒是否為負面
        negative_emotions = {"sad", "sadness", "angry", "anger", "fear", "disgust", "frustrated", "negative", "pain",
                             "worried"}
        is_negative = str(winner).lower() in negative_emotions if winner else False

        # 2. 獲取前一輪的負面狀態
        # 注意：這裡依賴歷史紀錄中是否有 'is_negative' 欄位
        is_previous_negative = self.history[-1].get("is_negative", False) if len(self.history) >= 1 else False

        # 3. 更新連續計數器
        if is_negative and is_previous_negative:
            # 如果本輪和前一輪都是負面，計數 +1 (代表連續 3 輪以上)
            self.worsening_count += 1
        elif is_negative:
            # 如果本輪是負面，但前一輪不是，則計數從 1 開始 (代表這是連續的第 1 輪)
            self.worsening_count = 1
        else:
            # 非負面情緒，重置計數
            self.worsening_count = 0

        # 4. 儲存本輪數據
        self.history.append({
            "text_emotion": text_emotion,
            "audio_emotion": audio_emotion,
            "facial_emotion": facial_emotion,
            "winner": winner,
            "is_negative": is_negative,  # <--- [新增] 紀錄本輪情緒極性
            "timestamp": time.time()
        })

    def get_dominant_emotion(self) -> str:
        if not self.history: return "neutral"
        # 使用 winner 作為 dominant emotion
        if self.history[-1].get("winner"): return self.history[-1]["winner"]
        return "neutral"

    def get_worsening_count(self) -> int:
        """回傳連續負面情緒的回合數 (2 代表連續兩輪，3 代表連續三輪...)"""
        return self.worsening_count

    def get_emotion_trend(self) -> str:
        if len(self.history) < 2: return "stable"
        negative_emotions = {"sad", "sadness", "angry", "anger", "fear", "disgust", "frustrated", "negative", "pain",
                             "worried"}
        positive_emotions = {"happy", "happiness", "joy", "positive", "excitement", "cheerful", "encouraging",
                             "surprised", "surprise"}
        scores = []
        for record in self.history:
            turn_score = 0
            winner = record.get("winner")
            if winner:
                w = str(winner).lower()
                if w in negative_emotions:
                    turn_score = -1
                elif w in positive_emotions:
                    turn_score = 1
            scores.append(turn_score)
        mid = len(scores) // 2
        if mid == 0: return "stable"
        older_avg = sum(scores[:mid]) / len(scores[:mid])
        recent_avg = sum(scores[mid:]) / len(scores[mid:])
        diff = recent_avg - older_avg
        if diff < -0.1:
            return "worsening"
        elif diff > 0.1:
            return "improving"
        else:
            if recent_avg <= -0.5:
                return "persistent_worsening"
            elif recent_avg >= 0.5:
                return "persistent_improving"
            else:
                return "stable"

# 輔助函數
def initialize_audio_settings(): pass


def get_audio_device_index(): return None


def play_audio(path): pass


def _audio_callback(*args): pass


def record_until_silence(*args): return None


def face_capture_worker(*args): pass


def clean_image_dir(): pass


def download_model(m, u, j, ju): pass


def has_wake_word(t): return config.WAKE_WORD in t.replace(" ", "")


def detect_language(text: str) -> str: return "zh" if any('\u4e00' <= c <= '\u9fff' for c in text) else "en"


def check_personality_switch_command(text: str, language: str) -> tuple:
    for pid, cfg in config.PERSONALITY_CONFIGS.items():
        if any(cmd in text.lower() for cmd in
               (cfg["switch_commands_zh"] if language == "zh" else cfg["switch_commands_en"])): return (True, pid)
    return (False, None)


def format_emotion_display(t, a, f, l): return ""