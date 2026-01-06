# model_loader.py
import warnings
import sys
import os
import torch
import webrtcvad
from faster_whisper import WhisperModel
from openai import OpenAI
# 移除 Piper，改用 CosyVoice
# from piper.voice import PiperVoice
from transformers import pipeline

# [新增] 匯入 FunASR 用於 Emotion2Vec+
from funasr import AutoModel

# 從本地模組匯入
import config
import utils
from facial_emotion_detector_pyfeat import initialize_facial_emotion_detector

warnings.filterwarnings("ignore", category=UserWarning)


class ModelManager:
    """一個專門用來載入和管理所有 AI 模型的類別"""

    def __init__(self):
        print("=" * 80)
        print("🚀 正在載入所有 AI 模型...")
        print("=" * 80)

        self.stt_model = self._load_stt()
        self.llm_client = self._connect_llm()

        # 修改：改為載入 CosyVoice 2.0
        self.cosvoice_model = self._load_tts_cosvoice()

        self.emotion_text_zh, self.emotion_text_en, self.emotion_audio_pipeline = self._load_emotion_models()
        self.facial_detector_enabled = self._load_facial_detector()

        # 載入 VAD
        self.vad = webrtcvad.Vad(config.VAD_AGGRESSIVENESS)
        print("✅ VAD 語音活動偵測已載入")

        # 獲取音訊裝置
        self.audio_device_index = utils.get_audio_device_index()

        print("\n" + "=" * 80)
        print("✨ 所有模型與裝置初始化完成！")
        print("=" * 80)

    def _load_stt(self):
        print("\n🧠 1/5 載入 Whisper STT 模型...")
        try:
            # 建議將 compute_type 改為 "float16" (如果有 GPU) 或維持 "int8"
            model = WhisperModel(config.WHISPER_MODEL_SIZE,
                                 device=config.WHISPER_DEVICE,
                                 compute_type=config.WHISPER_COMPUTE_TYPE)
            print("✅ Whisper 模型載入成功")
            return model
        except Exception as e:
            print(f"❌ Whisper 載入失敗: {e}")
            exit(1)

    def _connect_llm(self):
        print(f"\n🤖 2/5 連線本地 LLM API...")
        print(f"   端點: {config.LLM_BASE_URL}\n   模型: {config.LLM_MODEL}")
        try:
            client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
            # 嘗試列出模型以確認連線
            models = client.models.list()
            print("✅ 本地 LLM API 已連線")
            return client
        except Exception as e:
            print(f"❌ LLM 連線失敗: {e}\n   請確認 text-generation-webui 已啟動")
            exit(1)

    def _load_tts_cosvoice(self):
        """載入 CosyVoice 2.0 模型"""
        print("\n🔊 3/5 載入 CosyVoice 2.0 模型...")

        # 設定 CosyVoice 的路徑 (確保能 import 裡面的模組)
        cosyvoice_root = os.path.join(config.PROJECT_DIR, "CosyVoice")
        sys.path.append(cosyvoice_root)
        # CosyVoice 2 需要 Matcha-TTS
        sys.path.append(os.path.join(cosyvoice_root, "third_party", "Matcha-TTS"))

        if not torch.cuda.is_available():
            print("⚠️  未偵測到 GPU，CosyVoice 2 將無法運行或極慢！")

        try:
            # 動態匯入 CosyVoice2
            from cosyvoice.cli.cosyvoice import CosyVoice2

            # 模型路徑 (讀取 config 設定或使用預設)
            # 注意：這裡假設你已經下載模型到 CosyVoice/pretrained_models/CosyVoice2-0.5B
            model_dir = os.path.join(cosyvoice_root, config.COSVOICE_MODEL_DIR)

            print(f"   載入模型路徑: {model_dir}")

            # 初始化模型 (fp16=True 開啟半精度加速)
            model = CosyVoice2(model_dir, load_jit=False, load_trt=False, fp16=True)

            print("✅ CosyVoice 2.0 已就緒 (CUDA 加速中)")
            return model

        except ImportError as e:
            print(f"❌ 匯入失敗: {e}")
            print("   請確認已執行 pip install -r requirements.txt 並位於虛擬環境中")
            exit(1)
        except Exception as e:
            print(f"❌ CosyVoice 載入失敗: {e}")
            print(f"   請確認模型已下載至: {os.path.join(cosyvoice_root, config.COSVOICE_MODEL_DIR)}")
            exit(1)

    def _load_emotion_models(self):
        print("\n😊 4/5 載入情緒辨識模型(文字+語音)...")
        if not config.ENABLE_EMOTION_DETECTION:
            print("ℹ️  情緒辨識功能已停用")
            return None, None, None

        emotion_text_zh = None
        emotion_text_en = None
        emotion_audio_model = None
        try:
            print(f"   載入中文文字情緒模型: {config.EMOTION_TEXT_ZH_MODEL}")
            emotion_text_zh = pipeline("text-classification", model=config.EMOTION_TEXT_ZH_MODEL, top_k=None)
            print("   ✅ 中文文字情緒模型載入成功")

            print(f"   載入英文文字情緒模型: {config.EMOTION_TEXT_EN_MODEL}")
            emotion_text_en = pipeline("text-classification", model=config.EMOTION_TEXT_EN_MODEL, top_k=None)
            print("   ✅ 英文文字情緒模型載入成功")

            # [修改] 載入語音情緒模型 (Emotion2Vec+)
            # 注意：這裡不再鎖定版本號，讓它自動抓取最新版 (Base版適用)
            print(f"   載入語音情緒模型 (Emotion2Vec+): {config.EMOTION_AUDIO_MODEL}")
            emotion_audio_model = AutoModel(
                model=config.EMOTION_AUDIO_MODEL,
                trust_remote_code=True,
                disable_update=False
            )
            print("   ✅ 語音情緒辨識模型載入成功\n✅ 情緒辨識模型已就緒")

            return emotion_text_zh, emotion_text_en, emotion_audio_model

        except Exception as e:
            print(f"⚠️  情緒模型載入失敗: {e}")
            # 回傳 None 避免程式崩潰
            return emotion_text_zh, emotion_text_en, None

    def _load_facial_detector(self):
        print("\n🎭 5/5 載入 Py-FEAT 人臉情緒辨識模型...")
        if not config.ENABLE_FACIAL_EMOTION:
            print("ℹ️  人臉情緒辨識功能已停用")
            return False

        try:
            print(
                f"   模型配置:\n     • 人臉偵測: {config.PYFEAT_FACE_MODEL}\n     • 情緒模型: {config.PYFEAT_EMOTION_MODEL}\n     • 運算裝置: {config.PYFEAT_DEVICE.upper()}")

            # PyFeat 初始化
            success = initialize_facial_emotion_detector(
                face_model=config.PYFEAT_FACE_MODEL,
                emotion_model=config.PYFEAT_EMOTION_MODEL,
                device=config.PYFEAT_DEVICE
            )
            if success:
                print("✅ Py-FEAT 人臉情緒辨識模組已就緒")
                return True
            else:
                print("⚠️  Py-FEAT 初始化失敗，將跳過此功能")
                return False
        except Exception as e:
            print(f"⚠️  Py-FEAT 載入失敗: {e}\n   將跳過人臉情緒辨識功能")
            return False