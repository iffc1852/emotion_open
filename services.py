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
import time  # 🚀 [新增] 用於計時

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

# ==========================================

from zhconv import convert
from body_emotion_detector import BodyLanguageDetector
import queue
import threading
import sounddevice as sd

try:
    from cosyvoice.utils.file_utils import load_wav
except ImportError:
    print("❌ 錯誤: 依然找不到 CosyVoice 模組。")
    def load_wav(*args, **kwargs): return None

class AIServices:
    def __init__(self, models):
        self.models = models
        self.body_detector = BodyLanguageDetector()
        self.prompt_cache = {}
        self._preload_reference_audio()

    def _preload_reference_audio(self):
        try:
            if os.path.exists(config.COSVOICE_REFERENCE_WAV):
                wav_tensor = load_wav(str(config.COSVOICE_REFERENCE_WAV), 16000)
                self.prompt_cache["default"] = wav_tensor

            if hasattr(config, 'COSVOICE_VARIANTS'):
                for name, data in config.COSVOICE_VARIANTS.items():
                    wav_path = data.get("wav")
                    if wav_path and os.path.exists(wav_path):
                        wav_tensor = load_wav(str(wav_path), 16000)
                        self.prompt_cache[name] = wav_tensor
        except Exception as e:
            print(f"   ⚠️ 快取音訊失敗: {e}")

    def warm_up(self):
        print("   🔥 正在預熱 GPU 模型 (Warm-up)...")
        try:
            with torch.no_grad():
                dummy_text = convert("預熱", 'zh-cn')
                dummy_speech = self.prompt_cache.get("default")
                if dummy_speech is not None:
                    prompt_text = torch.zeros(1, 0, dtype=torch.int32)
                    self.models.cosvoice_model.inference_zero_shot(
                        dummy_text, prompt_text, dummy_speech, stream=True
                    )
            print("   ✅ GPU 預熱完成！隨時可以說話。")
        except Exception as e:
            print(f"   ⚠️ 預熱失敗: {e}")

    def transcribe(self, audio_data):
        """語音轉文字 (Whisper)"""
        if audio_data is None: return ""
        t_start = time.time()
        try:
            with torch.no_grad():
                audio_float = audio_data.astype(np.float32) / 32768.0
                segments, _ = self.models.stt_model.transcribe(
                    audio_float,
                    beam_size=1,
                    language="zh"
                    # 🚀 [刪除] 這裡拿掉了 initial_prompt，避免 AI 產生幻覺
                )
            text = "".join([segment.text for segment in segments])
            text = text.strip()

            # 🚀 [新增] 防呆過濾機制：如果有殘留的幻覺字眼，直接當作沒聽到
            if not text or "請以繁體中文" in text or "以下是一段普通話" in text or "字幕" in text:
                return ""

            duration = time.time() - t_start
            # 原本的 convert 已經會完美把簡體轉成繁體 (zh-tw) 了！
            return convert(text, 'zh-tw')

        except Exception as e:
            print(f"❌ STT 錯誤: {e}")
            return ""

    def detect_text_emotion(self, text, lang="zh"):
        if not text: return None
        t_start = time.time()
        try:
            with torch.no_grad():
                if hasattr(self.models, 'emotion_text_zh') and self.models.emotion_text_zh:
                    results = self.models.emotion_text_zh(text)
                    duration = time.time() - t_start
                    print(f"   ⏱️ [文字情緒] 耗時: {duration:.3f}s")

                    if not results: return None
                    top_result = results[0]
                    if isinstance(top_result, list): top_result = top_result[0]

                    raw_label = top_result['label']
                    score = top_result['score']

                    # 🚀 [修正] 處理 "难过/sad" 這種中英混合標籤
                    clean_label = str(raw_label)
                    if "/" in clean_label:
                        clean_label = clean_label.split("/")[-1].strip()

                    # 🚀 [修正] 查表防呆機制：同時比對「原字串」、「全大寫」、「全小寫」
                    # 這樣不管 config.py 裡面寫的是 "LABEL_1" 還是 "label_1"，都不會漏接了！
                    final_label = config.EMOTION_ZH_LABEL_MAP.get(clean_label,
                                                                  config.EMOTION_ZH_LABEL_MAP.get(clean_label.upper(),
                                                                                                  config.EMOTION_ZH_LABEL_MAP.get(
                                                                                                      clean_label.lower(),
                                                                                                      clean_label)))

                    return {"emotion": final_label, "score": score}
            return {"emotion": "neutral", "score": 0.5}
        except Exception as e:
            print(f"❌ 文字情緒錯誤: {e}")
            return {"emotion": "neutral", "score": 0.5}
    def detect_audio_emotion(self, audio_path, lang="zh"):
        t_start = time.time()
        try:
            if not hasattr(self.models, 'emotion_audio_pipeline') or self.models.emotion_audio_pipeline is None:
                return None

            with torch.no_grad():
                rec_result = self.models.emotion_audio_pipeline.generate(
                    input=audio_path, granularity="utterance", extract_embedding=False
                )

            duration = time.time() - t_start
            print(f"   ⏱️ [語音情緒] 耗時: {duration:.3f}s")

            if rec_result and len(rec_result) > 0:
                item = rec_result[0]
                if 'scores' in item and 'labels' in item:
                    scores = item['scores']
                    labels = item['labels']
                    top_idx = scores.index(max(scores))
                    return {"emotion": labels[top_idx], "score": scores[top_idx]}
            return None
        except Exception as e:
            print(f"❌ 語音情緒錯誤: {e}")
            return None

    def analyze_facial_emotion_from_images(self, images_data):
        if not images_data: return None
        t_start = time.time()
        target_frames = images_data[-1:]
        try:
            from facial_emotion_detector_pyfeat import analyze_frames_from_memory
            result = analyze_frames_from_memory(target_frames)

            duration = time.time() - t_start
            print(f"   ⏱️ [人臉情緒] 耗時: {duration:.3f}s")
            return result
        except ImportError:
            return None

    def detect_body_emotion(self, image_paths):
        t_start = time.time()
        try:
            result = self.body_detector.analyze_pose(image_paths)
            duration = time.time() - t_start
            print(f"   ⏱️ [肢體情緒] 耗時: {duration:.3f}s")
            return result
        except Exception as e:
            print(f"❌ 肢體分析失敗: {e}")
            return {"emotion": "neutral", "score": 0.0, "label": "無"}

    def generate_response(self, user_text, personality, history, text_emo, audio_emo, face_emo):
        """生成回應 (LLM)"""
        t_start = time.time()
        try:
            system_prompt = config.PERSONALITY_CONFIGS[personality]['prompt_prefix_zh']
            messages = [{"role": "system", "content": system_prompt}]
            for h in history[-4:]: messages.append(h)
            messages.append({"role": "user", "content": user_text})

            with torch.no_grad():
                completion = self.models.llm_client.chat.completions.create(
                    model=config.LLM_MODEL,
                    messages=messages,
                    max_tokens=config.LLM_MAX_TOKENS,
                    temperature=config.LLM_TEMPERATURE
                )
            reply = completion.choices[0].message.content

            duration = time.time() - t_start
            print(f"   ⏱️ [LLM 生成] 耗時: {duration:.3f}s")

            if reply:
                reply = re.sub(r'[\U00010000-\U0010ffff]', '', reply)
                reply = reply.replace("*", "").replace("（", "(").replace("）", ")")
                reply = reply.strip()
                reply = convert(reply, 'zh-tw')
                if not reply: reply = "嘿嘿，我都不知道該說什麼了！"

        except Exception as e:
            print(f"❌ LLM 生成失敗: {e}")
            reply = "抱歉，我現在有點秀逗。"

        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": reply})
        return reply, history

    def select_personality_auto(self, text_emo, audio_emo, face_emo, memory,
                                text_score=0, audio_score=0, face_score=0,
                                body_emo=None, body_score=0):
        try:
            import random
            weights = getattr(config, 'MODALITY_WEIGHTS', {"text": 0.3, "audio": 0.5, "facial": 0.2, "body": 0.0})
            emotion_scoreboard = {}

            audio_emotion_weights = getattr(config, 'EMOTION_AUDIO_WEIGHTS', {})
            # 🚀 [新增] 讀取 config 中的「最低信心度門檻」
            thresholds = getattr(config, 'EMOTION_CONFIDENCE_THRESHOLD', {"text": 0.4, "audio": 0.4, "facial": 0.4})

            # 修改函式，加入 modality_type 參數來辨識是哪種模態
            def vote_emotion(emotion, base_weight, score, modality_type):
                if not emotion: return

                # 🚀 [新增] 門檻守門員：如果分數低於設定值 (例如 0.4)，直接當作沒看到！
                min_req = thresholds.get(modality_type, 0.0)
                if score < min_req:
                    # print(f"   ⚠️ 忽略 {modality_type} 情緒: {emotion} (信心度 {score:.2f} 低於門檻 {min_req})")
                    return

                clean_emo = str(emotion).lower()
                if "/" in clean_emo:
                    clean_emo = clean_emo.split("/")[-1].strip()

                norm_emo = config.EMOTION_NORMALIZATION.get(clean_emo, clean_emo)

                multiplier = 1.0
                if modality_type == "audio":
                    multiplier = audio_emotion_weights.get(norm_emo, 1.0)

                final_score = base_weight * score * multiplier
                emotion_scoreboard[norm_emo] = emotion_scoreboard.get(norm_emo, 0.0) + final_score

            # 4. 投票時，標明自己的「模態名稱」，讓守門員檢查
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
            if isinstance(target_p, list):
                target_p = random.choice(target_p)

            print(f"   🤖 決定切換人格: {target_p}")
            return target_p

        except Exception as e:
            print(f"⚠️ 人格選擇邏輯錯誤: {e}，回退至預設")
            self.last_fused_emotion = "neutral"
            return "rational"

    def text_to_speech_cosvoice(self, text: str, output_path: str, personality: str) -> bool:
        t_start_func = time.time()
        try:
            from zhconv import convert
            prompt_speech_16k = self.prompt_cache.get(personality)
            if prompt_speech_16k is None:
                prompt_speech_16k = self.prompt_cache.get("default")
            if prompt_speech_16k is None: return False

            current_ref_text = getattr(config, 'COSVOICE_REFERENCE_TEXT', "")
            if hasattr(config, 'COSVOICE_VARIANTS') and personality in config.COSVOICE_VARIANTS:
                variant = config.COSVOICE_VARIANTS[personality]
                if variant.get("text"): current_ref_text = variant.get("text")

            prompt_text = torch.zeros(1, 0, dtype=torch.int32)
            if current_ref_text: prompt_text = convert(current_ref_text, 'zh-cn')

            clean_text = text.replace("- ", "，").replace("*", "").replace("\n", "。")
            final_input_text = convert(clean_text, 'zh-cn') if utils.detect_language(text) == "zh" else clean_text

            text_len = len(final_input_text)
            is_english = utils.detect_language(text) == "en"

            if is_english:
                start_threshold = 2 if text_len < 30 else (4 if text_len < 80 else 7)
            else:
                start_threshold = 2 if text_len < 10 else (4 if text_len < 30 else 7)

            t_inference_start = time.time()

            with torch.no_grad():
                output_gen = self.models.cosvoice_model.inference_zero_shot(
                    final_input_text, prompt_text, prompt_speech_16k, stream=True
                )

            target_sr = 22050
            buffer_queue = queue.Queue()

            def producer():
                first_chunk_generated = False
                try:
                    for i, res in enumerate(output_gen):
                        if not first_chunk_generated:
                            first_chunk_generated = True
                        chunk_tensor = res['tts_speech']
                        chunk_numpy = chunk_tensor.squeeze().cpu().numpy()
                        buffer_queue.put(chunk_numpy)
                    buffer_queue.put(None)
                except Exception as e:
                    buffer_queue.put(None)

            t = threading.Thread(target=producer)
            t.start()

            accumulated_chunks = []
            started_playing = False

            with sd.OutputStream(samplerate=target_sr, channels=1, dtype='float32') as stream:
                while True:
                    try:
                        timeout = 0.2 if started_playing else 5.0
                        chunk = buffer_queue.get(timeout=timeout)
                    except queue.Empty:
                        if t.is_alive(): continue
                        else: break

                    if chunk is None: break

                    if not started_playing:
                        accumulated_chunks.append(chunk)
                        if len(accumulated_chunks) >= start_threshold or not t.is_alive():
                            for c in accumulated_chunks: stream.write(c)
                            started_playing = True
                    else:
                        stream.write(chunk)
            t.join()
            return True
        except Exception as e:
            print(f"❌ 播放失敗: {e}")
            return False