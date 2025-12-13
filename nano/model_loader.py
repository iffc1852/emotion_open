# model_loader.py (Client 本地喚醒版)
import warnings
import webrtcvad
import whisper # 本地載入
import utils
import config
import torch
import logging

warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("whisper").setLevel(logging.WARNING)

class ModelManager:
    """在 Nano 上載入 VAD 和 Whisper STT 模組 (用於喚醒詞判斷)"""

    def __init__(self):
        print("=" * 80)
        print("🚀 正在載入 Nano Client (本地 WWD) 元件...")
        print("=" * 80)

        # 1. 載入 VAD
        self.vad = webrtcvad.Vad(config.VAD_AGGRESSIVENESS)
        print("✅ VAD 語音活動偵測已載入")

        # 2. 載入 Whisper (用於 WWD)
        self.stt_model = self._load_stt()
        
        # 載入 LLM/TTS/情緒模型 - 全部禁用
        self.llm_client = None 
        self.cosvoice_model = None
        self.emotion_text_zh = None
        self.emotion_text_en = None
        self.emotion_audio_pipeline = None
        self.facial_detector_enabled = False 

        self.audio_device_index = utils.get_audio_device_index()

        print("\n" + "=" * 80)
        print("✨ Client (本地 WWD) 初始化完成！")
        print("=" * 80)

    def _load_stt(self):
        """載入 Whisper STT 模組 (強制 CPU 模式)"""
        print(f"\n🧠 載入 Whisper STT 喚醒模型 ({config.WHISPER_MODEL_SIZE})...")
        try:
            # 使用 CPU 模式載入，保證成功
            model = whisper.load_model(config.WHISPER_MODEL_SIZE, device=config.WHISPER_DEVICE)
            print("✅ Whisper 模型載入成功 (CPU WWD 模式)")
            return model
        except Exception as e:
            print(f"❌ Whisper 載入失敗: {e}")
            exit(1)