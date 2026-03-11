# utils.py
import os
import time
import queue
import json
import wave
import shutil
import threading
import urllib.request
from datetime import datetime
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
import webrtcvad
import cv2
from PIL import Image
from pydub import AudioSegment
from pydub.utils import which

# 從 config 匯入設定
import config


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
        level_emoji = {"INFO": "ℹ️ ", "DEBUG": "🔍", "WARNING": "⚠️ ", "ERROR": "❌", "SUCCESS": "✅"}
        emoji = level_emoji.get(level, "  ")
        print(f"   [{timestamp}] {emoji} {message}")


# ===== 情緒記憶 =====

class EmotionMemory:
    def __init__(self, max_history=5):
        self.history = deque(maxlen=max_history)

    def add_emotion(self, text_emotion: str, audio_emotion: str, facial_emotion: str = None):
        self.history.append({
            "text_emotion": text_emotion, "audio_emotion": audio_emotion,
            "facial_emotion": facial_emotion, "timestamp": time.time()
        })

    def get_dominant_emotion(self) -> str:
        if not self.history: return "neutral"
        emotions = []
        for record in self.history:
            if record.get("facial_emotion"): emotions.append(record["facial_emotion"])
            if record.get("audio_emotion"): emotions.append(record["audio_emotion"])
            if record.get("text_emotion"): emotions.append(record["text_emotion"])
        if not emotions: return "neutral"
        return max(set(emotions), key=emotions.count)

    def get_emotion_trend(self) -> str:
        if len(self.history) < 2: return "stable"
        recent_emotions = []
        for r in self.history:
            emo = r.get("facial_emotion") or r.get("audio_emotion") or r.get("text_emotion")
            if emo: recent_emotions.append(emo)
        negative_emotions = {"sad", "sadness", "angry", "fear", "disgust", "negative", "frustrated"}
        positive_emotions = {"happy", "happiness", "joy", "positive", "excitement", "cheerful"}
        recent_score = sum(
            1 if e in positive_emotions else -1 if e in negative_emotions else 0 for e in recent_emotions[-2:])
        older_score = sum(
            1 if e in positive_emotions else -1 if e in negative_emotions else 0 for e in recent_emotions[:-2])
        if recent_score > older_score:
            return "improving"
        elif recent_score < older_score:
            return "worsening"
        return "stable"


# ===== 音訊裝置與播放 =====

def initialize_audio_settings():
    """設定音訊相關的全域參數"""
    sd.default.samplerate = config.SAMPLE_RATE
    sd.default.channels = config.CHANNELS

    ffmpeg_path = which("ffmpeg")
    if not ffmpeg_path:
        print("❌ 找不到 FFmpeg，請確認已加入 PATH");
        exit(1)
    AudioSegment.converter = ffmpeg_path

    config.TEMP_DIR.mkdir(exist_ok=True)

    print("✅ 音訊設定 (SoundDevice, FFmpeg) 初始化完成")


def get_audio_device_index():
    if config.AUDIO_DEVICE_INDEX is not None:
        print(f"   使用指定的音訊裝置: 索引 {config.AUDIO_DEVICE_INDEX}")
        return config.AUDIO_DEVICE_INDEX
    if not config.EXCLUDE_CAMERA_MIC:
        print(f"   使用系統預設音訊裝置")
        return None
    try:
        devices = sd.query_devices()
        camera_keywords = ['camera', 'webcam', 'usb video', 'integrated camera', 'facetime', 'hd camera', 'web cam',
                           'logitech', 'microsoft lifecam']

        # [刪除] 顯示掃描音訊裝置的訊息
        # print(f"\n🎤 掃描音訊輸入裝置...")
        # print(f"{'─' * 70}")

        input_devices = []
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                device_name = device['name'].lower()
                is_camera = any(keyword in device_name for keyword in camera_keywords)
                # status = "❌ 攝影機麥克風（已排除）" if is_camera else "✅ 可用"

                # [刪除] 顯示每個裝置的狀態
                # print(f"   [{i}] {device['name']}\n       狀態: {status}")

                if not is_camera:
                    input_devices.append((i, device['name']))

        # [刪除] 顯示分隔線
        # print(f"{'─' * 70}")

        if not input_devices:
            print(f"⚠️  未找到非攝影機的音訊裝置，將使用系統預設裝置")
            return None
        for idx, name in input_devices:
            name_lower = name.lower()
            if any(keyword in name_lower for keyword in ['microphone', 'mic', '麥克風', 'array']):
                print(f"✅ 自動選擇音訊裝置: [{idx}] {name}")
                return idx
        idx, name = input_devices[0]
        print(f"✅ 自動選擇音訊裝置: [{idx}] {name}")
        return idx
    except Exception as e:
        print(f"⚠️  音訊裝置掃描失敗: {e}\n   將使用系統預設裝置")
        return None


def play_audio(path: str):
    try:
        seg = AudioSegment.from_file(path)
        seg_resampled = seg.set_frame_rate(config.SAMPLE_RATE).set_channels(config.CHANNELS)
        samples = np.array(seg_resampled.get_array_of_samples()).astype(np.float32) / 32768.0
        print(f"🔊 播放中...");
        sd.play(samples, samplerate=config.SAMPLE_RATE, blocking=True);
        sd.stop()
    except Exception as e:
        print(f"❌ 播放失敗: {e}")


# ===== 錄音與 VAD =====

def _audio_callback(indata, frames_count, time_info, status, q):
    q.put(bytes(indata))


def record_until_silence(vad: webrtcvad.Vad, timeout: float, device_index=None, sync_face_capture: bool = False):
    q = queue.Queue()
    frames = []
    last_voice = time.time()

    prompt = f"🎙️  錄音中(靜音 > {timeout:.1f}s 結束)..."
    if sync_face_capture:
        prompt += " (同步採集臉部影像中)"
    print(prompt)

    with sd.RawInputStream(
            samplerate=config.SAMPLE_RATE, blocksize=config.FRAME_SIZE,
            dtype="int16", channels=config.CHANNELS,
            callback=lambda indata, frames_count, time_info, status: _audio_callback(indata, frames_count, time_info,
                                                                                     status, q),
            device=device_index
    ):
        while True:
            frame = q.get()
            if vad.is_speech(frame, config.SAMPLE_RATE):
                last_voice = time.time()
                frames.append(np.frombuffer(frame, dtype=np.int16))
            elif frames and (time.time() - last_voice) > timeout:
                break

    if len(frames) == 0:
        return None

    audio = np.concatenate(frames)
    print(f"   錄音完成，長度: {len(audio) / config.SAMPLE_RATE:.2f} 秒")
    return audio


# ===== 人臉擷取執行緒 =====

def face_capture_worker(stop_event, output_list, camera_index):
    debug_log(f"[Face Thread] 🎭 臉部採集執行緒啟動 (0.5FPS)...", "DEBUG")
    cap = None
    try:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            debug_log(f"[Face Thread] ❌ 錯誤：無法開啟攝影機 {camera_index}", "ERROR")
            return

        debug_log(f"[Face Thread] ℹ️ 攝影機暖機中...", "DEBUG")
        for _ in range(3):
            if stop_event.is_set(): break
            cap.read()

        base_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        frame_count = 0

        while not stop_event.is_set():
            start_of_capture = time.time()
            frame_count += 1
            debug_log(f"[Face Thread] 📷 正在採集第 {frame_count} 幀...", "DEBUG")

            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                debug_log(f"[Face Thread] ⚠️ 第 {frame_count} 幀讀取失敗", "WARNING")
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image_name = f"frame_{base_timestamp}_{frame_count}.jpg"
            image_path = config.IMAGE_DIR / image_name

            try:
                pil_image = Image.fromarray(frame_rgb)
                pil_image.save(str(image_path))
                output_list.append(str(image_path))
                debug_log(f"[Face Thread] ✅ 已儲存: {image_name}", "DEBUG")
            except Exception as e:
                debug_log(f"[Face Thread] ⚠️ 儲存第 {frame_count} 幀失敗: {e}", "ERROR")

            time_to_wait = 2.0 - (time.time() - start_of_capture)
            if time_to_wait > 0:
                stop_event.wait(time_to_wait)

    except Exception as e:
        debug_log(f"[Face Thread] ❌ 執行緒發生嚴重錯誤: {e}", "ERROR")
    finally:
        if cap:
            cap.release()
        debug_log(f"[Face Thread] 🛑 臉部採集執行緒停止。共採集 {len(output_list)} 幀。", "DEBUG")


# ===== 雜項工具 =====

def clean_image_dir():
    """啟動時清理並重建 image 資料夾"""
    print("\n🧹 正在清理舊的 image 資料夾...")
    if config.IMAGE_DIR.exists():
        try:
            shutil.rmtree(config.IMAGE_DIR)
            print(f"   ✅ 舊 {config.IMAGE_DIR.name} 資料夾已刪除。")
        except Exception as e:
            print(f"   ⚠️  刪除 {config.IMAGE_DIR.name} 資料夾失敗: {e}")

    try:
        config.IMAGE_DIR.mkdir(exist_ok=True)
        print(f"   ✅ 已建立新的 {config.IMAGE_DIR.name} 資料夾。")
    except Exception as e:
        print(f"   ❌ 建立 {config.IMAGE_DIR.name} 資料夾失敗: {e}")
        exit(1)


def download_model(model_name, model_url, json_name, json_url):
    model_path = config.MODEL_DIR / model_name
    json_path = config.MODEL_DIR / json_name
    try:
        if not model_path.exists():
            print(f"   > 正在下載模型: {model_name}...")
            urllib.request.urlretrieve(model_url, model_path)
            print("   ✅ 模型下載完成")
        else:
            print(f"   > 找到模型: {model_name}")
        if not json_path.exists():
            print(f"   > 正在下載設定檔: {json_name}...")
            urllib.request.urlretrieve(json_url, json_path)
            print("   ✅ 設定檔下載完成")
        else:
            print(f"   > 找到設定檔: {json_name}")
    except Exception as e:
        print(f"❌ 下載模型失敗: {e}")
        exit(1)


def has_wake_word(text: str) -> bool:
    return bool(text) and (config.WAKE_WORD in text.replace(" ", ""))


def detect_language(text: str) -> str:
    text_clean = ''.join(c for c in text if c.isalnum() or c.isspace())
    zh = sum(1 for c in text_clean if '\u4e00' <= c <= '\u9fff')
    en = sum(1 for c in text_clean if c.isalpha() and ord(c) < 128)
    if zh + en == 0: return "zh"
    return "zh" if zh / (zh + en) > 0.2 else "en"


def check_personality_switch_command(text: str, language: str) -> tuple:
    text_lower = text.lower()
    reset_commands = config.RESET_COMMANDS_ZH if language == "zh" else config.RESET_COMMANDS_EN
    for cmd in reset_commands:
        if cmd in text_lower: return (True, 'reset')
    for personality_id, cfg in config.PERSONALITY_CONFIGS.items():
        commands = cfg["switch_commands_zh"] if language == "zh" else cfg["switch_commands_en"]
        for cmd in commands:
            if cmd in text_lower: return (True, personality_id)
    return (False, None)


def format_emotion_display(text_emotion: dict, audio_emotion: dict, facial_emotion: dict, language: str) -> str:
    lines = ["📊 情緒分析結果:"]
    lines.append("─" * 70)
    if text_emotion:
        emotion = text_emotion['emotion'];
        emotion_zh = config.EMOTION_ZH_MAP.get(emotion, emotion)
        lines.append(f"   📝 文字情緒: {emotion_zh} ({emotion}) - 信心度: {text_emotion['score']:.2%}")
    if audio_emotion:
        emotion = audio_emotion.get('emotion', 'unknown');
        emotion_zh = config.EMOTION_ZH_MAP.get(emotion, emotion)
        lines.append(f"   🎤 語音情緒: {emotion_zh} ({emotion}) - 信心度: {audio_emotion.get('score', 0):.2%}")
    if facial_emotion:
        emotion_zh = facial_emotion.get('emotion_zh', 'unknown');
        emotion_en = facial_emotion.get('emotion', 'unknown')
        num_frames = facial_emotion.get('num_valid_frames', 1)
        total_frames = facial_emotion.get('num_total_frames', 1)
        lines.append(
            f"   🎭 臉部情緒: {emotion_zh} ({emotion_en}) - 信心度: {facial_emotion.get('confidence', 0):.2%} ({num_frames}/{total_frames} 幀平均)")
    return "\n".join(lines) if len(lines) > 2 else "   (無情緒資訊)"


# ==========================================
# 🚀 [新增] 填補詞播放功能
# ==========================================
def play_filler_randomly():
    """
    隨機播放一個填補音效 (非阻塞模式，不會卡住主程式)
    """
    # 1. 檢查設定檔是否開啟 (如果 config 沒這行，預設為 False)
    if not getattr(config, 'ENABLE_FILLER', True):
        return

    # 2. 檢查資料夾是否存在
    filler_dir = getattr(config, 'FILLER_AUDIO_DIR', config.PROJECT_DIR / "assets" / "fillers")
    if not filler_dir.exists():
        # 如果資料夾不存在，默默略過，不報錯
        return

    # 3. 找尋音檔 (.wav 或 .mp3)
    files = list(filler_dir.glob("*.wav")) + list(filler_dir.glob("*.mp3"))

    if files:
        import random
        import threading
        chosen = random.choice(files)

        # debug_log(f"🤔 播放填補音效: {chosen.name}", "DEBUG")

        # 4. 使用執行緒播放 (關鍵！這樣才不會卡住後面的運算)
        def _run():
            try:
                play_audio(str(chosen))  # 呼叫原本的 play_audio
            except Exception as e:
                print(f"⚠️ 填補詞播放失敗: {e}")

        threading.Thread(target=_run).start()