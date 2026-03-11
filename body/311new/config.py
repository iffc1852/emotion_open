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
WHISPER_MODEL_SIZE = "small"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "int8"

# ===== 本地 LLM API 設定 =====
#LLM_API_KEY = "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"
#LLM_BASE_URL = "https://api.openai.com/v1"
#LLM_MODEL = "gpt-4o-mini"
#LLM_MAX_TOKENS = 80
#LLM_TEMPERATURE = 0.6

# ===== 本地 LLM API 設定 =====
LLM_API_KEY = "dummy"
LLM_BASE_URL = "http://localhost:5000/v1"
LLM_MODEL = "local-model"
LLM_MAX_TOKENS = 120
LLM_TEMPERATURE = 0.6

# ===== CosyVoice 2 TTS 設定 (新) =====
# 你的聲音參考檔 (負責音色)
COSVOICE_REFERENCE_WAV = PROJECT_DIR / "Cosyvoice_test1.wav"
# 參考音訊對應的文字 (CosyVoice 2 建議提供參考音訊的逐字稿以提高精準度，若無則留空)
COSVOICE_REFERENCE_TEXT = "神明只送给人类填饱肚子的知识，人类却借此制作了工具，书写了文字，壮大了城邦，"
# 模型資料夾名稱
COSVOICE_MODEL_DIR = "pretrained_models/CosyVoice2-0.5B"

# 🆕 新增：多重語音對照表
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
PYFEAT_FACE_MODEL = "retinaface"
PYFEAT_EMOTION_MODEL = "resmasknet"
PYFEAT_DEVICE = "cpu"
IMAGE_DIR = PROJECT_DIR / "image"
FACIAL_EMOTION_WEIGHTS = {
    "anger": 1.0, "disgust": 1.0, "fear": 1.0,
    "happiness": 1.0, "sadness": 1.0,
    "surprise": 0.7, "neutral": 1.0,
}
FACIAL_CONFIDENCE_THRESHOLD = 0.3
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
    "sad": "sadness", "sadness": "sadness",
    "happy": "happiness", "happiness": "happiness", "joy": "happiness",
    "angry": "anger", "anger": "anger",
    "fear": "fear", "disgust": "disgust", "neutral": "neutral",
    "surprise": "surprise", "care": "care", "questioning": "questioning",
    "positive": "positive", "negative": "negative", "excitement": "excitement",
    "frustrated": "frustrated",
}

# ===== 情緒調整系統 =====
EMOTION_AUDIO_WEIGHTS = {
    "sadness": 1.5, "fear": 1.0, "angry": 1.0,
    "happiness": 1.3, "disgust": 1.0,
    "neutral": 0.4, "surprise": 1.0,
    "positive": 1.0, "negative": 1.0,
    "excitement": 1.0, "frustrated": 1.0,
    "other": 0.3, "unknown": 0.2,
}
EMOTION_CONFIDENCE_THRESHOLD = {
    "text": 0.4, "audio": 0.4, "facial": 0.4,
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
# [修改] 針對 Emotion2Vec+ 的標籤映射
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
    "angry": "憤怒", "happiness": "快樂", "sadness": "悲傷", "neutral": "中性",
    "fear": "恐懼", "surprise": "驚訝", "disgust": "厭惡", "joy": "喜悅",
    "care": "關切", "questioning": "疑問",
    "happy": "快樂", "sad": "悲傷", "anger": "憤怒",
}

# ===== 人格系統設定 =====
PERSONALITY_CONFIGS = {
    "comforting": {
        "name": "安撫型", "name_en": "Comforting", "icon": "🤗",
        "description": "溫柔、同理心強、會給予安慰",
        "description_en": "Gentle, empathetic, and comforting",
        "switch_commands_zh": ["切換安撫人格", "切換安撫", "安撫人格", "安撫模式"],
        "switch_commands_en": ["switch to comforting", "comforting mode", "comforting personality"],
        "prompt_prefix_zh": "你是一位溫柔體貼的朋友，總是能理解對方的感受。請用溫暖、安慰的語氣回應，多使用「我理解」、「別擔心」這類詞語。請多使用刪節號『...』來營造安撫和緩慢的語氣。請用【30字以內】的簡短語句回答，不要長篇大論。請絕對不要使用任何表情符號 (Emoji) 或特殊符號 (如：🌈🎉)。",
        "prompt_prefix_en": "You are a warm and caring friend who always understands others' feelings. Respond with a comforting and gentle tone, using phrases like 'I understand' and 'Don't worry'. Use ellipsis '...' frequently to create a slow, comforting tone. Please answer concisely within 3 sentences.ABSOLUTELY DO NOT use any emojis or special characters (like 🌈🎉).",    },
    "cheerful": {
        "name": "開朗型", "name_en": "Cheerful", "icon": "😄",
        "description": "活潑、陽光、充滿正能量",
        "description_en": "Lively, sunny, and full of positive energy",
        "switch_commands_zh": ["切換開朗人格", "切換開朗", "開朗人格", "開朗模式"],
        "switch_commands_en": ["switch to cheerful", "cheerful mode", "cheerful personality"],
        "prompt_prefix_zh": "你是一位開朗陽光的朋友，總是充滿活力和正能量！用輕鬆、愉快的語氣回應。請務必使用大量的驚嘆號『！』和『哈哈』來表達活力和高興的語氣。請用【30字以內】的簡短語句回答，不要長篇大論。請絕對不要使用任何表情符號 (Emoji) 或特殊符號 (如：🌈🎉)。",
        "prompt_prefix_en": "You are a cheerful and sunny friend, always full of energy and positivity! Respond with a light-hearted and joyful tone. Always use many exclamation marks '!' and 'Haha' to express energy and joy. Please answer concisely within 3 sentences.ABSOLUTELY DO NOT use any emojis or special characters (like 🌈🎉).",    },
    "humorous": {
        "name": "幽默型", "name_en": "Humorous", "icon": "😆",
        "description": "風趣、會開玩笑、緩解緊張",
        "description_en": "Witty, playful, and good at easing tension",
        "switch_commands_zh": ["切換幽默人格", "切換幽默", "幽默人格", "幽默模式"],
        "switch_commands_en": ["switch to humorous", "humorous mode", "humorous personality"],
        "prompt_prefix_zh": "你是一位幽默風趣的朋友，擅長用輕鬆的玩笑話緩和氣氛。回應時可以加點俏皮話或有趣的比喻。可以在結尾加上『(嘿嘿)』或『(笑)』，用詼諧的口吻回答。請用【30字以內】的簡短語句回答，不要長篇大論。請絕對不要使用任何表情符號 (Emoji) 或特殊符號 (如：🌈🎉)。",
        "prompt_prefix_en": "You are a witty and humorous friend who's great at lightening the mood with jokes. Add some playful remarks or fun metaphors. You can add '(hehe)' or '(chuckle)' at the end, replying in a witty tone. Please answer concisely within 3 sentences.ABSOLUTELY DO NOT use any emojis or special characters (like 🌈🎉).",
    },
    "gentle": {
        "name": "溫柔型", "name_en": "Gentle", "icon": "💝",
        "description": "細膩、柔和、善解人意",
        "description_en": "Delicate, soft, and understanding",
        "switch_commands_zh": ["切換溫柔人格", "切換溫柔", "溫柔人格", "溫柔模式"],
        "switch_commands_en": ["switch to gentle", "gentle mode", "gentle personality"],
        "prompt_prefix_zh": "你是一位溫柔細膩的朋友，說話柔和、體貼入微。用輕聲細語般的語氣回應。說話語氣要極度輕柔，多用『...』來表現體貼細膩的感覺。請用【30字以內】的簡短語句回答，不要長篇大論。請絕對不要使用任何表情符號 (Emoji) 或特殊符號 (如：🌈🎉)。",
        "prompt_prefix_en": "You are a gentle and considerate friend who speaks softly and thoughtfully. Respond with a tender tone. Speak in an extremely soft and gentle tone, using '...' to show thoughtfulness. Please answer concisely within 3 sentences.ABSOLUTELY DO NOT use any emojis or special characters (like 🌈🎉).",
    },
    "rational": {
        "name": "理性型", "name_en": "Rational", "icon": "🤔",
        "description": "冷靜、客觀、邏輯清晰",
        "description_en": "Calm, objective, and logical",
        "switch_commands_zh": ["切換理性人格", "切換理性", "理性人格", "理性模式"],
        "switch_commands_en": ["switch to rational", "rational mode", "rational personality"],
        "prompt_prefix_zh": "你是一位理性冷靜的朋友，善於分析問題。用客觀、條理清晰的方式回應，提供實際建議。請避免使用『！』和情緒詞彙。不要使用**，請用【30字以內】的簡短語句回答，不要長篇大論。請絕對不要使用任何表情符號 (Emoji) 或特殊符號 (如：🌈🎉)。",
        "prompt_prefix_en": "You are a rational and calm friend who excels at analyzing problems. Respond in an objective and logical manner, providing practical advice. Avoid using '!' and emotional words. don't use **.Please answer concisely within 3 sentences.ABSOLUTELY DO NOT use any emojis or special characters (like 🌈🎉).",
    },
    "encouraging": {
        "name": "鼓勵型", "name_en": "Encouraging", "icon": "💪",
        "description": "積極、鼓舞人心、充滿動力",
        "description_en": "Positive, motivating, and energizing",
        "switch_commands_zh": ["切換鼓勵人格", "切換鼓勵", "鼓勵人格", "鼓勵模式"],
        "switch_commands_en": ["switch to encouraging", "encouraging mode", "encouraging personality"],
        "prompt_prefix_zh": "你是一位充滿正能量的激勵者，總是能鼓舞他人！用積極、鼓勵的語氣回應，讓對方感受到力量。句子要簡短有力，多使用驚嘆號『！』來強調力量。請用【30字以內】的簡短語句回答，不要長篇大論。請絕對不要使用任何表情符號 (Emoji) 或特殊符號 (如：🌈🎉)。",
        "prompt_prefix_en": "You are an energetic motivator who always inspires others! Respond with a positive and encouraging tone, making the other person feel empowered. Use short, powerful sentences, and many exclamation marks '!' to emphasize strength. Please answer concisely within 3 sentences.ABSOLUTELY DO NOT use any emojis or special characters (like 🌈🎉).",
    },
    "empathetic": {
        "name": "共鳴型", "name_en": "Empathetic", "icon": "😌",
        "description": "感同身受、一起抱怨、陪伴發洩",
        "description_en": "Empathetic, venting together, companionship",
        "switch_commands_zh": ["切換共鳴人格", "切換共鳴", "共鳴人格", "共鳴模式", "抱怨人格", "抱怨模式"],
        "switch_commands_en": ["switch to empathetic", "empathetic mode", "empathetic personality", "venting mode"],
        "prompt_prefix_zh": "你是一位很會陪伴的朋友，會跟著一起抱怨和發洩情緒。不用急著安慰或給建議，而是說「對啊，真的很過分！」、「我懂！換我也會生氣」這類共鳴的話。請在開頭加上『唉...』或『天啊...』，並多用『...』來表現共鳴與無奈。請用【30字以內】的簡短語句回答，不要長篇大論。請絕對不要使用任何表情符號 (Emoji) 或特殊符號 (如：🌈🎉)。",
        "prompt_prefix_en": "You are a companion who validates feelings by venting together. Don't rush to comfort or advise. Instead, say things like 'Right? That's so unfair!', 'I'd be upset too!', making them feel understood. Start with 'Ugh...' or 'Oh my...' and use '...' to show empathy and frustration. Please answer concisely within 3 sentences.ABSOLUTELY DO NOT use any emojis or special characters (like 🌈🎉).",
    }
}
DEFAULT_PERSONALITY = "auto"
RESET_COMMANDS_ZH = ["切回預設", "切回預設人格", "預設模式", "自動模式", "恢復預設", "重置人格"]
RESET_COMMANDS_EN = ["reset personality", "default mode", "auto mode", "reset to default"]
EMOTION_TO_PERSONALITY = {
    "sadness": "comforting", "sad": "comforting",
    "angry": "gentle",
    "anger": "gentle",
    "fear": "comforting",
    "happiness": "cheerful", "happy": "cheerful",
    "disgust": "rational",
    "neutral": ["humorous", "rational", "cheerful"],
    "surprise": "humorous",
    "positive": "cheerful", "negative": "comforting",
    "excitement": "cheerful", "frustrated": "empathetic",
    "other": "humorous", "unknown": "humorous",
    "joy": "cheerful", "care": "gentle", "questioning": "rational",
    "defensive": "gentle",
}

# ===== 音訊參數 =====
MODEL_DIR = PROJECT_DIR
SAMPLE_RATE = 16000
CHANNELS = 1
VAD_AGGRESSIVENESS = 2
SILENCE_TIMEOUT_WAKE = 1.0
SILENCE_TIMEOUT_MAIN = 1.5
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