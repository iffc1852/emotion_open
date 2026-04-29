# services.py
import os
import sys
import re
import cv2
import torch
import torchaudio
import numpy as np
import config
import utils
import time

# ==========================================
# 強制加入 CosyVoice 路徑
# ==========================================
try:
    cosyvoice_root = str(config.PROJECT_DIR / "CosyVoice")
    matcha_root = str(config.PROJECT_DIR / "CosyVoice" / "third_party" / "Matcha-TTS")
    if cosyvoice_root not in sys.path:
        sys.path.append(cosyvoice_root)
    if matcha_root not in sys.path:
        sys.path.append(matcha_root)
except Exception as e:
    print(f"⚠️ 路徑設定警告: {e}")

from zhconv import convert
import queue
import threading
import sounddevice as sd

try:
    from cosyvoice.utils.file_utils import load_wav
except ImportError:
    print("❌ 錯誤: 依然找不到 CosyVoice 模組。")

    def load_wav(*args, **kwargs):
        return None

class AIServices:
    def __init__(self, models):
        self.models = models
        self.prompt_cache = {}
        self._preload_reference_audio()

    def _preload_reference_audio(self):
        """🚀 從 config.COSVOICE_VARIANTS 預載所有音色"""
        try:
            # 載入全域預設
            if hasattr(config, 'COSVOICE_REFERENCE_WAV') and os.path.exists(config.COSVOICE_REFERENCE_WAV):
                self.prompt_cache["default"] = load_wav(str(config.COSVOICE_REFERENCE_WAV), 16000)

            # 載入多重人格音訊
            if hasattr(config, 'COSVOICE_VARIANTS'):
                for name, data in config.COSVOICE_VARIANTS.items():
                    wav_path = data.get("wav")  # 這裡精準對應您的 config 標籤
                    if wav_path and os.path.exists(wav_path):
                        self.prompt_cache[name] = load_wav(str(wav_path), 16000)
                        # print(f"   ✅ 已成功快取音色: {name}")
        except Exception as e:
            print(f"   ⚠️ 快取音訊失敗: {e}")

    def warm_up(self):
        """🔥 消滅首句延遲：大腦與嘴巴同步暖機"""
        print("   🔥 正在預熱 GPU 模型 (Warm-up)...")
        # LLM 預熱
        try:
            messages = [{"role": "user", "content": "你好"}]
            self.models.llm_client.chat.completions.create(model=config.LLM_MODEL, messages=messages, max_tokens=5)
            print("      ✅ LLM 大腦預熱完成！")
        except Exception:
            pass

        # TTS 預熱 (使用 FP32 穩定版)
        try:
            with torch.inference_mode():
                dummy_speech = self.prompt_cache.get("default")
                if dummy_speech is not None:
                    self.models.cosvoice_model.inference_zero_shot(
                        convert("預熱", 'zh-cn'), torch.zeros(1, 0, dtype=torch.int32), dummy_speech, stream=False
                    )
            print("      ✅ CosyVoice 嘴巴預熱完成！")
        except Exception:
            pass

    def transcribe(self, audio_data, force_language=None):
        if audio_data is None: return ""
        t_start = time.time()
        try:
            audio_float = audio_data.astype(np.float32) / 32768.0

            transcribe_kwargs = {
                "beam_size": 3,  # 速度優先。如果覺得辨識變差，可以改 1~5
                "language": force_language,
                "task": "transcribe",
                "vad_filter": True,
                "vad_parameters": dict(min_silence_duration_ms=500),
                "condition_on_previous_text": False,  # 降低 Whisper 胡思亂想與幻覺
            }

            # 只有強制中文時才加中文提示，避免之後英文對話被影響
            if force_language == "zh":
                transcribe_kwargs["initial_prompt"] = "這是一段繁體中文的日常對話。"

            segments, info = self.models.stt_model.transcribe(
                audio_float,
                **transcribe_kwargs
            )


            text = "".join([segment.text for segment in segments]).strip()

            if not text or "請以繁體中文" in text or "以下是一段" in text:
                return ""

            print(f"   [🗣️ STT 聽打] (偵測語言: {info.language}) 耗時: {time.time() - t_start:.2f}s ➔ 「{text}」")

            if info.language == "zh" or utils.detect_language(text) == "zh":
                return convert(text, 'zh-tw')

            return text


        except Exception as e:
            print(f"❌ STT 錯誤: {e}")
            return ""


    def detect_text_emotion(self, text, lang="zh"):
        if not text: return None
        try:
            with torch.inference_mode():
                if hasattr(self.models, 'emotion_text_zh') and self.models.emotion_text_zh:
                    results = self.models.emotion_text_zh(text)
                    if not results: return None
                    top_result = results[0]
                    if isinstance(top_result, list): top_result = top_result[0]

                    raw_label = top_result['label']
                    score = top_result['score']

                    clean_label = str(raw_label)
                    if "/" in clean_label: clean_label = clean_label.split("/")[-1].strip()

                    final_label = config.EMOTION_ZH_LABEL_MAP.get(clean_label,
                                                                  config.EMOTION_ZH_LABEL_MAP.get(clean_label.upper(),
                                                                                                  config.EMOTION_ZH_LABEL_MAP.get(
                                                                                                      clean_label.lower(),
                                                                                                      clean_label)))

                    return {"emotion": final_label, "score": score}
            return {"emotion": "neutral", "score": 0.5}
        except Exception as e:
            return {"emotion": "neutral", "score": 0.5}

    def detect_audio_emotion(self, audio_path, lang="zh"):
        try:
            if not hasattr(self.models,
                           'emotion_audio_pipeline') or self.models.emotion_audio_pipeline is None: return None
            with torch.inference_mode():
                rec_result = self.models.emotion_audio_pipeline.generate(
                    input=audio_path,
                    granularity="utterance",
                    extract_embedding=False
                )

            if rec_result and len(rec_result) > 0:
                item = rec_result[0]
                if 'scores' in item and 'labels' in item:
                    scores = item['scores']
                    labels = item['labels']
                    top_idx = scores.index(max(scores))
                    raw_label = labels[top_idx]
                    mapped_label = getattr(config, 'EMOTION_AUDIO_LABEL_MAP', {}).get(raw_label, raw_label)

                    return {"emotion": mapped_label, "score": scores[top_idx]}

            return None
        except Exception as e:
            return None

    def analyze_facial_emotion_from_images(self, images_data):
        if not images_data: return None
        target_frames = images_data[-2:] #最後看幾張圖
        try:
            from facial_emotion_detector_pyfeat import analyze_frames_from_memory
            return analyze_frames_from_memory(target_frames)
        except ImportError:
            return None

    def detect_body_emotion(self, image_paths):
        return {"emotion": "neutral", "score": 0.0, "label": "無"}

    def select_personality_auto(self, text_emo, audio_emo, face_emo, memory, text_score=0, audio_score=0, face_score=0,
                                body_emo=None, body_score=0):
        try:
            import random
            weights = getattr(config, 'MODALITY_WEIGHTS', {"text": 0.3, "audio": 0.5, "facial": 0.2, "body": 0.0})
            emotion_scoreboard = {}

            thresholds = getattr(config, 'EMOTION_CONFIDENCE_THRESHOLD', {
                "text": 0.4,
                "audio": 0.4,
                "facial": 0.4
            })

            def vote_emotion(emotion, base_weight, score, modality_type):
                if not emotion:
                    return

                min_req = thresholds.get(modality_type, 0.0)
                if score < min_req:
                    return

                clean_emo = str(emotion).lower()

                if "/" in clean_emo:
                    clean_emo = clean_emo.split("/")[-1].strip()

                norm_emo = config.EMOTION_NORMALIZATION.get(clean_emo, clean_emo)

                # 注意：
                # 情緒分數已經在 gui_main.py 或 facial_emotion_detector_pyfeat.py 加權過一次。
                # 這裡只乘「模態權重」，不要再乘情緒權重。
                final_score = base_weight * score

                emotion_scoreboard[norm_emo] = emotion_scoreboard.get(norm_emo, 0.0) + final_score

            vote_emotion(text_emo, weights.get("text", 0.3), text_score, "text")
            vote_emotion(audio_emo, weights.get("audio", 0.5), audio_score, "audio")
            vote_emotion(face_emo, weights.get("facial", 0.2), face_score, "facial")
            vote_emotion(body_emo, weights.get("body", 0.0), body_score, "body")

            if not emotion_scoreboard:
                final_emotion = "neutral"
            else:
                final_emotion = max(emotion_scoreboard, key=emotion_scoreboard.get)

            print(f"   📊 當前綜合情緒結算: {final_emotion} (計分板: {emotion_scoreboard})")
            self.last_fused_emotion = final_emotion

            target_p = config.EMOTION_TO_PERSONALITY.get(final_emotion, "rational")
            if isinstance(target_p, list): target_p = random.choice(target_p)
            print(f"   🤖 決定切換人格: {target_p}")
            return target_p
        except Exception as e:
            self.last_fused_emotion = "neutral"
            return "rational"

        #  在參數最後面加上 language="zh"
    def generate_response(self, user_text, personality, history, text_emo, audio_emo, face_emo, language="zh"):
            """ 單次完整生成 (非串流)，避免 GPU 互鎖"""

        # 🚀 根據語言選擇 Prompt
            if language == "en":
                base_prompt = getattr(config, 'GLOBAL_SYSTEM_PROMPT_EN', "")
                personality_prompt = config.PERSONALITY_CONFIGS[personality].get('prompt_prefix_en', "")
                system_prompt = f"{base_prompt}\n\n[Personality Setup]\n{personality_prompt}" if base_prompt else personality_prompt
            else:
                base_prompt = getattr(config, 'GLOBAL_SYSTEM_PROMPT_ZH', "")
                personality_prompt = config.PERSONALITY_CONFIGS[personality].get('prompt_prefix_zh', "")
                system_prompt = f"{base_prompt}\n\n【當前人格設定】\n{personality_prompt}" if base_prompt else personality_prompt

            messages = [{"role": "system", "content": system_prompt}]
            for h in history[-4:]: messages.append(h)
            messages.append({"role": "user", "content": user_text})

            reply = "抱歉，我現在有點秀逗。" if language == "zh" else "Sorry, I'm having a little glitch."
            try:
                with torch.no_grad():
                    completion = self.models.llm_client.chat.completions.create(
                        model=config.LLM_MODEL,
                        messages=messages,
                        max_tokens=config.LLM_MAX_TOKENS,
                        temperature=config.LLM_TEMPERATURE,
                        stream=False
                    )
                reply = completion.choices[0].message.content
                if reply:
                    reply = re.sub(r'[\U00010000-\U0010ffff]', '', reply)
                    reply = reply.replace("*", "").replace("（", "(").replace("）", ")").strip()
                    # 🚀 只有中文才做繁體轉換
                    if language == "zh":
                        reply = convert(reply, 'zh-tw')
            except Exception as e:
                print(f"❌ LLM 生成錯誤: {e}")

            return reply, history

            # ==========================================
            #  [全新核心] 流水線專用函式 (雙語支援版)
            # ==========================================

    def generate_response_stream(self, user_text, personality, history, language="zh"):
        """LLM 文字串流生成 (邊想邊出字)"""

        #  2. 根據語言切換 System Prompt 與人格設定
        if language == "en":
            base_prompt = getattr(config, 'GLOBAL_SYSTEM_PROMPT_EN', "")
            # 使用 .get() 避免 config 裡沒有設定到英文 prompt 時報錯
            personality_prompt = config.PERSONALITY_CONFIGS[personality].get('prompt_prefix_en', "")
            system_prompt = f"{base_prompt}\n\n[Personality Setup]\n{personality_prompt}" if base_prompt else personality_prompt
        else:
            base_prompt = getattr(config, 'GLOBAL_SYSTEM_PROMPT_ZH', "")
            personality_prompt = config.PERSONALITY_CONFIGS[personality].get('prompt_prefix_zh', "")
            system_prompt = f"{base_prompt}\n\n【當前人格設定】\n{personality_prompt}" if base_prompt else personality_prompt

        messages = [{"role": "system", "content": system_prompt}]
        for h in history[-10:]: messages.append(h)
        messages.append({"role": "user", "content": user_text})

        try:
            # 🚀 使用 stream=True 並設定更短的 timeout
            completion = self.models.llm_client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=messages,
                stream=True,
                timeout=5  # 強制縮短連線等待
            )

            for chunk in completion:
                #  關鍵：只要 WebUI 噴出一個碎片，就立刻 yield 出去
                token = chunk.choices[0].delta.content
                if token:
                    yield token

        except Exception as e:
            print(f"❌ API 傳輸阻塞: {e}")


    def synthesize_cosvoice_audio(self, text: str, personality: str):
        """ 穩定版語音合成：確保文字與音色 100% 對齊"""
        t_start = time.time()
        try:
            # 1. 抓取音色張量
            prompt_speech_16k = self.prompt_cache.get(personality, self.prompt_cache.get("default"))
            if prompt_speech_16k is None: return None, 0.0

            # 2.  [關鍵修復] 從 COSVOICE_VARIANTS 抓取對應的 text
            current_ref_text = getattr(config, 'COSVOICE_REFERENCE_TEXT', "")
            if hasattr(config, 'COSVOICE_VARIANTS') and personality in config.COSVOICE_VARIANTS:
                # 這裡精準對應您的 config 裡的 "text" 欄位
                current_ref_text = config.COSVOICE_VARIANTS[personality].get("text", current_ref_text)

            prompt_text = torch.zeros(1, 0, dtype=torch.int32)
            if current_ref_text:
                prompt_text = convert(current_ref_text, 'zh-cn')

            # 3. 準備待合成文字
            clean_text = text.replace("*", "").replace("\n", "。")
            final_input_text = convert(clean_text, 'zh-cn')

            # 4. 推論 (不使用 autocast，確保 FP32 絕對穩定)
            with torch.inference_mode():
                output_gen = self.models.cosvoice_model.inference_zero_shot(
                    final_input_text, prompt_text, prompt_speech_16k, stream=False
                )
                result = next(output_gen)
                audio_data = result['tts_speech'].squeeze().cpu().numpy()

            return audio_data, time.time() - t_start
        except Exception as e:
            print(f"⚠️ TTS 錯誤: {e}")
            return None, 0.0