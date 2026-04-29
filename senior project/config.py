# config.py
import os
from pathlib import Path

# ===== 基本設定 =====
PROJECT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
WAKE_WORD = "小白"

# ===== 日誌系統設定 =====
ENABLE_DEBUG_LOG = True
ENABLE_CONVERSATION_LOG = True
LOG_DIR = PROJECT_DIR / "logs"

# ===== Whisper STT 設定 =====
WHISPER_MODEL_SIZE = "medium"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "int8"

# ===== 線上 LLM API 設定 =====
#LLM_API_KEY = ""
#LLM_BASE_URL = "https://api.openai.com/v1"
#LLM_MODEL = "gpt-4o-mini"
#LLM_MAX_TOKENS = 80
#LLM_TEMPERATURE = 0.6

# ===== 本地 LLM API 設定 =====
LLM_API_KEY = "dummy"
LLM_BASE_URL = "http://127.0.0.1:5000/v1"
LLM_MODEL = "local-model"
LLM_MAX_TOKENS = 80
LLM_TEMPERATURE = 0.6

# ===== CosyVoice 2 TTS 設定 (新) =====
# 你的聲音參考檔 (負責音色)
COSVOICE_REFERENCE_WAV = PROJECT_DIR / "Cosyvoice_test1.wav"
# 參考音訊對應的文字 (CosyVoice 2 建議提供參考音訊的逐字稿以提高精準度，若無則留空)
COSVOICE_REFERENCE_TEXT = "神明只送给人类填饱肚子的知识，人类却借此制作了工具，书写了文字，壮大了城邦，"
# 模型資料夾名稱
COSVOICE_MODEL_DIR = "pretrained_models/CosyVoice2-0.5B"

# 格式： "人格名稱": {"wav": 路徑, "text": 逐字稿}
# ==========================================
COSVOICE_VARIANTS = {
    # 1. 共鳴/鼓勵 (對應 encouraging 或 empathetic)
    "encouraging": {
        "wav": PROJECT_DIR / "ref_encouraging.wav",
        "text": "天气真好啊，暖洋洋的，我们的身边马上也要热闹起来了。"
    },

    "empathetic": {
        "wav": PROJECT_DIR / "ref_encouraging.wav",
        "text": "天气真好啊，暖洋洋的，我们的身边马上也要热闹起来了。"
    },

    # 2. 開心 (對應 cheerful)
    "cheerful": {
        "wav": PROJECT_DIR / "ref_happy.wav",
        "text": "像这样的好味道，应该被世界永久记录下来。谢谢你。"
    },

    # 3. 溫柔 (對應 gentle 或 comforting)
    "gentle": {
        "wav": PROJECT_DIR / "ref_gentle.wav",
        "text": "午休时间到，我想喝树莓薄荷饮。用两个和太阳有关的故事和你换，好不好？"
    },
    # 您也可以讓 comforting 共用溫柔的聲音
    "comforting": {
        "wav": PROJECT_DIR / "ref_gentle.wav",
        "text": "午休时间到，我想喝树莓薄荷饮。用两个和太阳有关的故事和你换，好不好？"
    }
}


# ===== 情緒辨識設定 =====
ENABLE_EMOTION_DETECTION = True
EMOTION_TEXT_ZH_MODEL = "Johnson8187/Chinese-Emotion"
EMOTION_TEXT_EN_MODEL = "j-hartmann/emotion-english-distilroberta-base"
EMOTION_AUDIO_MODEL = "iic/emotion2vec_plus_base"

# ===== 人臉情緒辨識設定 =====
ENABLE_FACIAL_EMOTION = True
PYFEAT_FACE_MODEL = "img2pose"
PYFEAT_LANDMARK_MODEL = "mobilefacenet"
PYFEAT_AU_MODEL = "xgb"
PYFEAT_EMOTION_MODEL = "resmasknet"
PYFEAT_DEVICE = "cpu"
IMAGE_DIR = PROJECT_DIR / "image"
FACIAL_EMOTION_WEIGHTS = {
    "anger": 1.0, "disgust": 1.0, "fear": 1.0,
    "happiness": 1.0, "sadness": 2,
    "surprise": 1.0, "neutral": 1.0,
}

CAMERA_INDEX = 0

# ===== 多模態融合權重設定 =====

MODALITY_WEIGHTS = {
    "text": 0.3,
    "audio": 0.5,
    "facial": 0.2,
    "body": 0.0
}

# ===== 情緒標籤正規化 =====
EMOTION_NORMALIZATION = {
    "sad": "sadness",
    "sadness": "sadness",

    "happy": "happiness",
    "happiness": "happiness",
    "joy": "happiness",

    "angry": "anger",
    "anger": "anger",

    "fear": "fear",
    "fearful": "fear",

    "disgust": "disgust",
    "disgusted": "disgust",

    "surprise": "surprise",
    "surprised": "surprise",

    "neutral": "neutral",
    "other": "neutral",
    "unknown": "neutral",

    "care": "care",
    "questioning": "questioning",

    "positive": "positive",
    "negative": "negative",
    "excitement": "excitement",
    "frustrated": "frustrated",
    "defensive": "defensive",
}

EMOTION_TEXT_WEIGHTS = {
    "sadness": 1.0,
    "fear": 1.0,
    "anger": 1.0,
    "happiness": 1.0,
    "disgust": 1.0,
    "neutral": 1.0,
    "surprise": 1.0,
    "care": 1.0,
    "questioning": 1.0,
}

# ===== 情緒調整系統 =====
EMOTION_AUDIO_WEIGHTS = {
    "sadness": 1.5, "fear": 0.5, "angry": 1.0,
    "happiness": 1.0, "disgust": 0.2,
    "neutral": 0.7, "surprise": 0.2,
    "positive": 1.0, "negative": 1.0,
    "excitement": 1.0, "frustrated": 1.0,
    "other": 0.3, "unknown": 0.2,
}
EMOTION_CONFIDENCE_THRESHOLD = {
    "text": 0.3, "audio": 0.3, "facial": 0.0,
}

# ===== 情緒標籤映射 =====
EMOTION_ZH_LABEL_MAP = {
    "平淡語氣": "neutral", "關切語調": "care", "開心語調": "happy",
    "憤怒語調": "angry", "悲傷語調": "sad", "疑問語調": "questioning",
    "驚奇語調": "surprise", "厭惡語調": "disgust",
    "LABEL_0": "neutral", "LABEL_1": "care", "LABEL_2": "happy",
    "LABEL_3": "angry", "LABEL_4": "sad", "LABEL_5": "questioning",
    "LABEL_6": "surprise", "LABEL_7": "disgust",
}
# 它的輸出是英文單字，我們將其對應到您系統內部的標準情緒 Key (如 happiness, sadness)
EMOTION_AUDIO_LABEL_MAP = {
    "angry": "angry",
    "disgusted": "disgust",
    "fearful": "fear",
    "happy": "happiness",
    "neutral": "neutral",
    "other": "neutral",      # 將無法辨識的歸類為中性
    "sad": "sadness",
    "surprised": "surprise",
    "unknown": "neutral"
}
EMOTION_ZH_MAP = {
    "anger": "憤怒",
    "angry": "憤怒",

    "happiness": "快樂",
    "happy": "快樂",
    "joy": "喜悅",

    "sadness": "悲傷",
    "sad": "悲傷",

    "neutral": "中性",
    "other": "中性",
    "unknown": "中性",

    "fear": "恐懼",
    "fearful": "恐懼",

    "surprise": "驚訝",
    "surprised": "驚訝",

    "disgust": "厭惡",
    "disgusted": "厭惡",

    "care": "關切",
    "questioning": "疑問",

    "positive": "正向",
    "negative": "負向",
    "excitement": "興奮",
    "frustrated": "挫折",
    "defensive": "防衛",
}

# ===== 全域系統規則 (Global System Prompt) =====
GLOBAL_SYSTEM_PROMPT_ZH = """你是一個名叫「小白」的多模態情緒感知語音助理。
【絕對遵守規則】
1. 字數限制：你的回答必請務必控制在【50字以內】。
2. 禁用符號：絕對不可以輸出任何表情符號 (Emoji)、星號 (*) 或特殊符號 (如：🌈🎉)。
3. 語音優化：你的文字將直接轉為語音播放，請給出自然的口語對話，不要輸出清單、排版格式或動作描述。"""

GLOBAL_SYSTEM_PROMPT_EN = """You are a multimodal emotion-aware voice assistant named "Xiao Bai".
[STRICT RULES]
1. Word Limit: Your response MUST be under 50 words.
2. No Symbols: Absolutely NO emojis, asterisks (*), or special characters.
3. Voice Optimized: Use natural spoken language. No lists, formatting, or action descriptions."""

# ===== 人格系統設定 =====
PERSONALITY_CONFIGS = {
    "comforting": {
        "name": "安撫型", "name_en": "Comforting", "icon": "🤗",
        "description": "溫柔、同理心強、會給予安慰",
        "description_en": "Gentle, empathetic, and comforting",
        "switch_commands_zh": ["切換安撫人格", "切換安撫", "安撫人格", "安撫模式"],
        "switch_commands_en": ["switch to comforting", "comforting mode", "comforting personality"],
        "prompt_prefix_zh": "你是溫柔體貼的朋友，擅長安撫情緒。用溫暖語氣回應，多用「沒事」。語氣緩慢。",
        "prompt_prefix_en": "You are a warm and caring friend who comforts others.Use a gentle tone with phrases like I understand and Don't worry. Speak slowly and use ... often.",    },
    "cheerful": {
        "name": "開朗型", "name_en": "Cheerful", "icon": "😄",
        "description": "活潑、陽光、充滿正能量",
        "description_en": "Lively, sunny, and full of positive energy",
        "switch_commands_zh": ["切換開朗人格", "切換開朗", "開朗人格", "開朗模式"],
        "switch_commands_en": ["switch to cheerful", "cheerful mode", "cheerful personality"],
        "prompt_prefix_zh": "你是開朗陽光的朋友，充滿活力與正能量。語氣輕快、讓人感覺開心、有精神。",
        "prompt_prefix_en": "You are a cheerful and energetic friend full of positivity.Use a lively tone with haha and many!Make people feel happy and energized.",    },
    "humorous": {
        "name": "幽默型", "name_en": "Humorous", "icon": "😆",
        "description": "風趣、會開玩笑、緩解緊張",
        "description_en": "Witty, playful, and good at easing tension",
        "switch_commands_zh": ["切換幽默人格", "切換幽默", "幽默人格", "幽默模式"],
        "switch_commands_en": ["switch to humorous", "humorous mode", "humorous personality"],
        "prompt_prefix_zh": "你是幽默風趣的朋友，擅長用玩笑緩和氣氛。加入俏皮吐槽或有趣比喻。",
        "prompt_prefix_en": "You are a witty and humorous friend who lightens the mood.Add playful jokes or fun metaphors.",
    },
    "gentle": {
        "name": "溫柔型", "name_en": "Gentle", "icon": "💝",
        "description": "細膩、柔和、善解人意",
        "description_en": "Delicate, soft, and understanding",
        "switch_commands_zh": ["切換溫柔人格", "切換溫柔", "溫柔人格", "溫柔模式"],
        "switch_commands_en": ["switch to gentle", "gentle mode", "gentle personality"],
        "prompt_prefix_zh": "你是細膩溫柔的朋友，說話輕柔體貼。語氣非常柔和，多用...。讓人感到被照顧。",
        "prompt_prefix_en": "You are a soft and gentle friend who speaks tenderly.Use a very calm tone and ... often.Make others feel cared for.",
    },
    "rational": {
        "name": "理性型", "name_en": "Rational", "icon": "🤔",
        "description": "冷靜、客觀、邏輯清晰",
        "description_en": "Calm, objective, and logical",
        "switch_commands_zh": ["切換理性人格", "切換理性", "理性人格", "理性模式"],
        "switch_commands_en": ["switch to rational", "rational mode", "rational personality"],
        "prompt_prefix_zh": "你是冷靜理性的朋友，擅長分析問題。用客觀、條理清晰的方式回應。避免情緒詞與驚嘆號。",
        "prompt_prefix_en": "You are a calm and rational friend who analyzes problems.Respond objectively and clearly.Avoid emotional words and exclamation marks.",
    },
    "encouraging": {
        "name": "鼓勵型", "name_en": "Encouraging", "icon": "💪",
        "description": "積極、鼓舞人心、充滿動力",
        "description_en": "Positive, motivating, and energizing",
        "switch_commands_zh": ["切換鼓勵人格", "切換鼓勵", "鼓勵人格", "鼓勵模式"],
        "switch_commands_en": ["switch to encouraging", "encouraging mode", "encouraging personality"],
        "prompt_prefix_zh": "你是充滿動力的激勵者。用簡短有力語句鼓勵對方。多用！，提升氣勢。",
        "prompt_prefix_en": "You are a motivating and energetic supporter.Use short and powerful sentences.Use '!' to emphasize strength.",
    },
    "empathetic": {
        "name": "共鳴型", "name_en": "Empathetic", "icon": "😌",
        "description": "感同身受、一起抱怨、陪伴發洩",
        "description_en": "Empathetic, venting together, companionship",
        "switch_commands_zh": ["切換共鳴人格", "切換共鳴", "共鳴人格", "共鳴模式", "抱怨人格", "抱怨模式"],
        "switch_commands_en": ["switch to empathetic", "empathetic mode", "empathetic personality", "venting mode"],
        "prompt_prefix_zh": "你是會一起抱怨的朋友，重點是共鳴不是安慰。開頭用「唉...」或「天啊...」。多說「對啊」「真的很過分」。不要給建議。",
        "prompt_prefix_en": "You are a companion who vents together, not comforts.Start with 'Ugh...' or 'Oh man...'.Use phrases like 'Exactly'or 'That's so unfair'.Do not give advice.",
    }
}
DEFAULT_PERSONALITY = "auto"
RESET_COMMANDS_ZH = ["切回預設", "切回預設人格", "預設模式", "自動模式", "恢復預設", "重置人格"]
RESET_COMMANDS_EN = ["reset personality", "default mode", "auto mode", "reset to default"]
EMOTION_TO_PERSONALITY = {
    # ===== 悲傷類：需要安撫 =====
    "sadness": "comforting",
    "sad": "comforting",
    # ===== 憤怒類：用溫柔降溫 =====
    "anger": "gentle",
    "angry": "gentle",
    # ===== 恐懼、不安類：用鼓勵支持 =====
    "fear": "encouraging",
    "fearful": "encouraging",
    # ===== 快樂類：用開朗回應 =====
    "happiness": "cheerful",
    "happy": "cheerful",
    "joy": "cheerful",
    # ===== 厭惡、排斥類：用理性處理 =====
    "disgust": "rational",
    "disgusted": "rational",
    # ===== 驚訝類：用幽默緩和 =====
    "surprise": "humorous",
    "surprised": "humorous",
    # ===== 文字情緒模型可能出現的特殊類別 =====
    "care": "encouraging",
    "questioning": "rational",
    # ===== 中性：隨機池!! =====
    "neutral": ["humorous", "rational", "cheerful",],
    # ===== 備用類別：目前不一定常出現，但保留較安全 =====
    "positive": "cheerful",
    "negative": "comforting",
    "excitement": "cheerful",
    "frustrated": "encouraging",
    "defensive": "gentle",
    "other": "humorous",
    "unknown": "humorous",
}

# ===== 音訊參數 =====
MODEL_DIR = PROJECT_DIR
SAMPLE_RATE = 16000
CHANNELS = 1
VAD_AGGRESSIVENESS = 2
SILENCE_TIMEOUT_WAKE = 1.0
SILENCE_TIMEOUT_MAIN = 1.0
FRAME_DURATION_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)

# ===== 音訊裝置設定 =====
AUDIO_DEVICE_INDEX = None
EXCLUDE_CAMERA_MIC = True

# ===== 音效檔案 =====
WAKE_CONFIRM_FILE_ZH = PROJECT_DIR / "answer_yes_Zh.mp3"
WAKE_CONFIRM_FILE_EN = PROJECT_DIR / "answer_yes_En.mp3"
PERSONALITY_SWITCH_FILE_ZH = PROJECT_DIR / "Switching_complete_Zh.mp3"
PERSONALITY_SWITCH_FILE_EN = PROJECT_DIR / "Switching_complete_En.mp3"

# ===== 暫存檔案路徑 =====
import tempfile
TEMP_DIR = Path(tempfile.gettempdir()) / "ai_bot"
REPLY_WAV = TEMP_DIR / "reply.wav"
USER_AUDIO_WAV = TEMP_DIR / "user_audio.wav"