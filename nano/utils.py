# utils.py (修正版 - 移除 cv2 依賴)
import os
import time
import json
import shutil
import subprocess
import threading
import queue
import sys
import re
from datetime import datetime
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torchaudio
import noisereduce as nr
import sounddevice as sd
# import cv2 # <--- 修正: 移除 cv2 匯入
import webrtcvad

import config

# ===== 1. 智慧裝置偵測 (保持不變) =====

def get_usb_audio_card_id():
    """... (函式內容保持不變) ..."""
    try:
        # 執行 Linux 指令列出裝置
        result = subprocess.run(['aplay', '-l'], capture_output=True, text=True)
        # 逐行分析
        for line in result.stdout.split('\n'):
            if "card" in line and "USB" in line:
                match = re.search(r'card\s+(\d+)', line)
                if match:
                    card_id = match.group(1)
                    return card_id
    except Exception as e:
        print(f"⚠️ 無法偵測音效卡: {e}")
    return None

def get_audio_device_index():
    """
    自動尋找 USB 麥克風的 sounddevice 索引 (Input)
    """
    # 1. 如果 config 有指定數字，就優先用 config 的
    if config.AUDIO_DEVICE_INDEX is not None:
        try:
            # 嘗試查詢固定的裝置索引
            sd.query_devices(config.AUDIO_DEVICE_INDEX)
            return config.AUDIO_DEVICE_INDEX
        except Exception as e:
            # 2. 如果查詢固定索引失敗，嘗試強制重新初始化
            print(f"⚠️ 查詢索引 {config.AUDIO_DEVICE_INDEX} 失敗，嘗試強制重啟音訊驅動... 錯誤: {e}")
            try:
                # 強制底層重啟 (解決 ALSA 鎖定問題)
                sd._terminate() 
                sd._initialize() 
                # 再次查詢一次
                sd.query_devices(config.AUDIO_DEVICE_INDEX)
                return config.AUDIO_DEVICE_INDEX
            except Exception:
                 print("❌ 強制重啟後仍失敗，回退到自動偵測...")
                 # 繼續執行下方的自動偵測邏輯...
                 pass # 繼續執行下方的自動偵測邏輯

    # 3. 自動偵測邏輯 (當固定索引失敗或設為 None 時執行)
    try:
        devices = sd.query_devices()
        # 1. 優先尋找名字裡有 'USB' 的麥克風
        for i, dev in enumerate(devices):
            if 'USB' in dev['name'] and dev['max_input_channels'] > 0:
                print(f"🎤 自動鎖定 USB 麥克風: {dev['name']} (Index: {i})")
                return i
        
        # 2. 如果沒 USB，找任何能錄音的
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                if "tegra" in dev['name'].lower() or "ape" in dev['name'].lower():
                    continue 
                print(f"🎤 使用系統麥克風: {dev['name']} (Index: {i})")
                return i
                
        print("⚠️ 找不到可用的麥克風裝置！")
        return None
    except Exception as e:
        print(f"❌ 麥克風搜尋失敗: {e}")
        return None


# ===== 2. 核心音訊處理 (保持不變) =====

def load_wav_to_numpy(path: str, sample_rate: int) -> tuple[np.ndarray | None, str | None]:
    """... (函式內容保持不變) ..."""
    try:
        audio_path = Path(path)
        raw_wav_path = audio_path.with_suffix('.wav')

        command = [
            "ffmpeg", "-i", str(audio_path), "-ar", str(sample_rate), "-ac", "1", "-y", str(raw_wav_path)
        ]

        if os.name == 'nt':
            subprocess.run(command, capture_output=True, text=True, startupinfo=subprocess.STARTUPINFO())
        else:
            subprocess.run(command, capture_output=True, text=True)

        waveform, sr = torchaudio.load(str(raw_wav_path))

        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        audio_np_raw = waveform.squeeze().numpy()
        cleaned_audio = audio_np_raw

        if len(audio_np_raw) > sample_rate * 0.1:
            try:
                cleaned_audio = nr.reduce_noise(y=audio_np_raw, sr=sample_rate, stationary=True, prop_decrease=0.90)
            except Exception:
                pass

        waveform_stt = torch.from_numpy(cleaned_audio).unsqueeze(0)
        max_val = torch.abs(waveform_stt).max()
        if max_val > 0:
            scale_factor = 0.9 / max_val
            waveform_stt = waveform_stt * scale_factor

        audio_np_final = waveform_stt.squeeze().numpy()
        return audio_np_final, str(raw_wav_path)

    except Exception as e:
        print(f"❌ load_wav_to_numpy 失敗: {e}")
        return None, None


# ===== 3. 錄音與播放 (保持不變) =====

def initialize_audio_settings():
    """... (函式內容保持不變) ..."""
    try:
        sd.query_devices()
    except Exception as e:
        print(f"⚠️ 音訊裝置初始化失敗: {e}")

def record_until_silence(vad, silence_timeout, device_index=None, sync_face_capture=False):
    """... (函式內容保持不變) ..."""
    FRAME_DURATION_MS = 30
    SAMPLE_RATE = 16000
    CHUNK_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
    
    q = queue.Queue()

    def audio_callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    audio_buffer = []
    silence_frames = 0
    has_voice_started = False
    max_silence_frames = int(silence_timeout * 1000 / FRAME_DURATION_MS)
    ring_buffer = deque(maxlen=5) 
    
    print("   ... (等待說話) ...", end="\r")

    try:
        # 如果 device_index 是 None，讓 get_audio_device_index 再找一次
        if device_index is None:
            device_index = get_audio_device_index()

        with sd.InputStream(samplerate=SAMPLE_RATE, blocksize=CHUNK_SIZE,
                            device=device_index, channels=1, dtype='int16',
                            callback=audio_callback):
            while True:
                chunk = q.get()
                audio_buffer.append(chunk)

                is_speech = vad.is_speech(chunk.tobytes(), SAMPLE_RATE)
                ring_buffer.append(is_speech)
                is_speech_smooth = sum(ring_buffer) > (ring_buffer.maxlen / 2)

                if is_speech_smooth:
                    if not has_voice_started:
                        print("   🎙️  偵測到語音，開始錄製...     ")
                        has_voice_started = True
                    silence_frames = 0
                elif has_voice_started:
                    silence_frames += 1
                
                if has_voice_started and silence_frames > max_silence_frames:
                    print("   🛑 說話結束 (靜音偵測)          ")
                    break
                
                # 簡單的防呆：如果一直沒說話，避免 buffer 爆掉，每隔一段時間清空未觸發的 buffer
                if not has_voice_started and len(audio_buffer) > 300: # 約 10秒
                    audio_buffer = audio_buffer[-10:] # 只保留最後一點點

    except Exception as e:
        print(f"\n❌ 錄音發生錯誤: {e}")
        return None

    if not audio_buffer: return None
    full_audio = np.concatenate(audio_buffer, axis=0).flatten()
    return full_audio

def play_audio(path):
    """... (函式內容保持不變) ..."""
    if not os.path.exists(path):
        print(f"❌ 找不到音訊檔案: {path}")
        return

    try:
        # 1. 自動偵測 USB 喇叭的 Card ID
        card_id = get_usb_audio_card_id()
        
        # 2. 決定播放參數
        device_arg = f"plughw:{card_id},0" if card_id else "default"
        
        if str(path).endswith(".wav"):
            # 使用 aplay
            cmd = ["aplay", "-q", str(path)]
            if card_id:
                cmd.extend(["-D", device_arg])
            subprocess.run(cmd, check=False)
        else:
            # 使用 ffplay
            env = os.environ.copy()
            if card_id:
                env["AUDIODEV"] = device_arg
            subprocess.run(["ffplay", "-nodisp", "-autoexit", "-hide_banner", str(path)], 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env, check=False)

    except Exception as e:
        print(f"❌ 播放失敗: {e}")


# ===== 4. 影像處理 (全部移除，只保留空函式和清理) =====
# 由於 ENABLE_FACIAL_EMOTION = False，這些函式不會被 main.py 呼叫
# 但為了代碼完整性，我們移除 cv2 依賴並將函式留空

def face_capture_worker(stop_event, image_paths, camera_index=0):
    """(已移除人臉辨識，此為空函式)"""
    return

def clean_image_dir():
    """(保持清理函式)"""
    if config.IMAGE_DIR.exists(): shutil.rmtree(config.IMAGE_DIR)
    config.IMAGE_DIR.mkdir(parents=True, exist_ok=True)


# ===== 5. 輔助邏輯 (保持不變) =====

class ConversationLogger:
    """... (類別內容保持不變) ..."""
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
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
        return {"total_turns": len(self.session_log)} # 簡化版

class EmotionMemory:
    """... (類別內容保持不變) ..."""
    def __init__(self, max_history=10):
        self.history = deque(maxlen=max_history)
        self.worsening_count = 0

    def add_emotion(self, text_emotion: str, audio_emotion: str, facial_emotion: str = None, winner: str = None):
        negative_emotions = {"sad", "sadness", "angry", "anger", "fear", "disgust", "frustrated", "negative", "pain", "worried"}
        is_negative = str(winner).lower() in negative_emotions if winner else False
        is_previous_negative = self.history[-1].get("is_negative", False) if len(self.history) >= 1 else False

        if is_negative and is_previous_negative: self.worsening_count += 1
        elif is_negative: self.worsening_count = 1
        else: self.worsening_count = 0

        self.history.append({
            "text_emotion": text_emotion, "audio_emotion": audio_emotion,
            "facial_emotion": facial_emotion, "winner": winner,
            "is_negative": is_negative, "timestamp": time.time()
        })

    def get_dominant_emotion(self) -> str:
        if not self.history: return "neutral"
        if self.history[-1].get("winner"): return self.history[-1]["winner"]
        return "neutral"

    def get_worsening_count(self) -> int: return self.worsening_count

    def get_emotion_trend(self) -> str: return "stable" # 簡化版邏輯，完整版請保留原本的

def debug_log(message: str, level: str = "INFO"):
    if config.ENABLE_DEBUG_LOG:
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"   [{timestamp}] {level} {message}")

def has_wake_word(text: str) -> bool:
    if not text: return False
    clean_text = text.lower().replace(" ", "").replace(",", "").replace("，", "")
    return config.WAKE_WORD.lower() in clean_text

def detect_language(text: str) -> str:
    if not text: return "en"
    return "zh" if any('\u4e00' <= c <= '\u9fff' for c in text) else "en"

def check_personality_switch_command(text: str, language: str) -> tuple:
    if not text: return (False, None)
    text_lower = text.lower()
    reset_cmds = config.RESET_COMMANDS_ZH if language == "zh" else config.RESET_COMMANDS_EN
    if any(cmd in text_lower for cmd in reset_cmds): return (True, 'reset')
    for pid, cfg in config.PERSONALITY_CONFIGS.items():
        cmds = cfg["switch_commands_zh"] if language == "zh" else cfg["switch_commands_en"]
        if any(cmd in text_lower for cmd in cmds): return (True, pid)
    return (False, None)

def format_emotion_display(text_res, audio_res, facial_res, language):
    output = []
    if text_res: output.append(f"📝 文字: {text_res['emotion']} ({text_res['score']:.2%})")
    if audio_res: output.append(f"🎙️ 語音: {audio_res['emotion']} ({audio_res['score']:.2%})")
    if facial_res: output.append(f"😐 人臉: {facial_res['emotion']} ({facial_res['confidence']:.2%})")
    return " | ".join(output)

# utils.py 檔案結尾處，新增以下函式：

def download_and_save_audio(url, save_path):
    """從伺服器下載 TTS 音訊檔"""
    import requests
    try:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except requests.exceptions.RequestException as e:
        debug_log(f"❌ TTS 音訊下載失敗: {e}", "ERROR")
        return False