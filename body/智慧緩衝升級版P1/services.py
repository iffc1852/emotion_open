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


    def load_wav(*args, **kwargs):
        return None


class AIServices:
    def __init__(self, models):
        self.models = models
        self.body_detector = BodyLanguageDetector()

        # 🚀 [優化] 參考音訊快取 (Cache)
        self.prompt_cache = {}
        self._preload_reference_audio()

    def _preload_reference_audio(self):
        """預先載入所有角色的參考音訊到 RAM"""
        try:
            # 1. 載入預設音訊
            if os.path.exists(config.COSVOICE_REFERENCE_WAV):
                wav_tensor = load_wav(str(config.COSVOICE_REFERENCE_WAV), 16000)
                self.prompt_cache["default"] = wav_tensor

            # 2. 載入其他人格音訊
            if hasattr(config, 'COSVOICE_VARIANTS'):
                for name, data in config.COSVOICE_VARIANTS.items():
                    wav_path = data.get("wav")
                    if wav_path and os.path.exists(wav_path):
                        wav_tensor = load_wav(str(wav_path), 16000)
                        self.prompt_cache[name] = wav_tensor
        except Exception as e:
            print(f"   ⚠️ 快取音訊失敗: {e}")

    def warm_up(self):
        """🚀 [優化] 模型預熱"""
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
        t_start = time.time()  # ⏱️
        try:
            with torch.no_grad():
                audio_float = audio_data.astype(np.float32) / 32768.0
                segments, _ = self.models.stt_model.transcribe(
                    audio_float,
                    beam_size=1,
                    language="zh"
                )
            text = "".join([segment.text for segment in segments])

            duration = time.time() - t_start
            # print(f"   ⏱️ [Whisper STT] 耗時: {duration:.3f}s") # 若覺得太洗版可註解

            return convert(text.strip(), 'zh-tw')

        except Exception as e:
            print(f"❌ STT 錯誤: {e}")
            return ""

    def detect_text_emotion(self, text, lang="zh"):
        if not text: return None
        t_start = time.time()  # ⏱️
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
                    final_label = config.EMOTION_ZH_LABEL_MAP.get(raw_label, raw_label)
                    return {"emotion": final_label, "score": score}
            return {"emotion": "neutral", "score": 0.5}
        except Exception as e:
            print(f"❌ 文字情緒錯誤: {e}")
            return {"emotion": "neutral", "score": 0.5}

    def detect_audio_emotion(self, audio_path, lang="zh"):
        t_start = time.time()  # ⏱️
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
        t_start = time.time()  # ⏱️
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
        t_start = time.time()  # ⏱️
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
        t_start = time.time()  # ⏱️
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

                # 強制轉繁體顯示
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
            import random  # 確保有引入隨機模組

            candidates = {}

            # 1. 讀取並顯示權重
            if hasattr(config, 'MODALITY_WEIGHTS'):
                weights = config.MODALITY_WEIGHTS
                print(
                    f"   ⚖️ 當前權重設定: 文字={weights.get('text')}, 語音={weights.get('audio')}, 人臉={weights.get('facial')}, 肢體={weights.get('body')}")
            else:
                print("   ⚠️ 警告: config.py 缺少 MODALITY_WEIGHTS，使用預設平均值！")
                weights = {"text": 0.25, "audio": 0.25, "facial": 0.25, "body": 0.25}

            def vote(emotion, weight, score):
                if not emotion: return
                norm_emo = config.EMOTION_NORMALIZATION.get(emotion, emotion)

                # 取得對應人格
                target_p = config.EMOTION_TO_PERSONALITY.get(norm_emo)

                # 🎲 [關鍵修正] 如果拿到的是列表，立刻抽籤變成字串！
                if isinstance(target_p, list):
                    # print(f"   🎲 從 {target_p} 中隨機抽選...") # debug用
                    target_p = random.choice(target_p)

                if target_p:
                    # 累積該人格的分數
                    candidates[target_p] = candidates.get(target_p, 0.0) + weight * score

            # 2. 開始投票
            vote(text_emo, weights.get("text", 0.3), text_score)
            vote(audio_emo, weights.get("audio", 0.3), audio_score)
            vote(face_emo, weights.get("facial", 0.2), face_score)
            vote(body_emo, weights.get("body", 0.2), body_score)

            # 3. 慣性加分 (上一輪的情緒)
            dominant_prev = memory.get_dominant_emotion()
            if dominant_prev and dominant_prev != "neutral":
                norm_prev = config.EMOTION_NORMALIZATION.get(dominant_prev, dominant_prev)
                target_prev = config.EMOTION_TO_PERSONALITY.get(norm_prev)

                # 慣性也要支援列表隨機
                if isinstance(target_prev, list):
                    target_prev = random.choice(target_prev)

                if target_prev and target_prev in candidates:
                    candidates[target_prev] += 0.05

            # 4. 結算
            if not candidates:
                # 如果沒人投票 (全部棄權)，從 neutral 清單隨機抽一個
                default_p = config.EMOTION_TO_PERSONALITY.get("neutral", "rational")
                if isinstance(default_p, list):
                    print(f"   🎲 無明確情緒，從中性池隨機選擇...")
                    return random.choice(default_p)
                return default_p

            best_personality = max(candidates, key=candidates.get)
            print(f"   📊 戰況: 勝出 -> {best_personality}")
            return best_personality
        except Exception as e:
            print(f"⚠️ 人格選擇邏輯錯誤: {e}，回退至預設")
            return "rational"

    def text_to_speech_cosvoice(self, text: str, output_path: str, personality: str) -> bool:
        """(TTS) CosyVoice 2.0 - 快取 + 動態緩衝 + no_grad + 計時"""
        t_start_func = time.time()  # ⏱️ 函式呼叫開始時間
        try:
            from zhconv import convert

            prompt_speech_16k = self.prompt_cache.get(personality)
            if prompt_speech_16k is None:
                prompt_speech_16k = self.prompt_cache.get("default")

            if prompt_speech_16k is None:
                print("❌ 錯誤: 無法取得參考音訊 (快取為空)")
                return False

            current_ref_text = getattr(config, 'COSVOICE_REFERENCE_TEXT', "")
            if hasattr(config, 'COSVOICE_VARIANTS') and personality in config.COSVOICE_VARIANTS:
                variant = config.COSVOICE_VARIANTS[personality]
                if variant.get("text"): current_ref_text = variant.get("text")

            prompt_text = torch.zeros(1, 0, dtype=torch.int32)
            if current_ref_text: prompt_text = convert(current_ref_text, 'zh-cn')

            clean_text = text.replace("- ", "，").replace("*", "").replace("\n", "。")

            # 轉簡體 (為了 CosyVoice 準確度)
            final_input_text = convert(clean_text, 'zh-cn') if utils.detect_language(text) == "zh" else clean_text

            # 動態緩衝
            text_len = len(final_input_text)
            is_english = utils.detect_language(text) == "en"

            if is_english:
                if text_len < 30:
                    start_threshold = 2
                elif text_len < 80:
                    start_threshold = 4
                else:
                    start_threshold = 7
            else:
                if text_len < 10:
                    start_threshold = 2
                elif text_len < 30:
                    start_threshold = 4
                else:
                    start_threshold = 7

            utils.debug_log(f"TTS 生成: {text_len}字 ({'EN' if is_english else 'ZH'}), 緩衝{start_threshold}段",
                            "DEBUG")

            # ⏱️ 開始推論計時
            t_inference_start = time.time()

            with torch.no_grad():
                output_gen = self.models.cosvoice_model.inference_zero_shot(
                    final_input_text, prompt_text, prompt_speech_16k, stream=True
                )

            # 注意: inference_zero_shot 回傳 generator 很快，真正的計算在下面 loop

            target_sr = 22050
            buffer_queue = queue.Queue()

            def producer():
                first_chunk_generated = False
                try:
                    for i, res in enumerate(output_gen):
                        # ⏱️ 記錄第一段生成的真正時間 (首字延遲)
                        if not first_chunk_generated:
                            latency = time.time() - t_inference_start
                            print(f"   ⏱️ [TTS 首字延遲] 耗時: {latency:.3f}s")
                            first_chunk_generated = True

                        chunk_tensor = res['tts_speech']
                        chunk_numpy = chunk_tensor.squeeze().cpu().numpy()
                        buffer_queue.put(chunk_numpy)
                    buffer_queue.put(None)

                    # ⏱️ 全部生成完畢時間
                    total_gen_time = time.time() - t_inference_start
                    # print(f"   ⏱️ [TTS 全部生成] 耗時: {total_gen_time:.3f}s") # 可選開啟

                except Exception as e:
                    print(f"❌ 生成錯誤: {e}")
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
                        if t.is_alive():
                            if started_playing: print(".", end="", flush=True)
                            continue
                        else:
                            break

                    if chunk is None: break

                    if not started_playing:
                        accumulated_chunks.append(chunk)
                        if len(accumulated_chunks) >= start_threshold or not t.is_alive():
                            # 實際播放不計入生成時間，但會顯示何時開始播
                            print(f"   🔊 緩衝完畢(囤貨{len(accumulated_chunks)}段)，開始播放...")
                            for c in accumulated_chunks:
                                stream.write(c)
                            started_playing = True
                    else:
                        stream.write(chunk)
            t.join()

            return True
        except Exception as e:
            print(f"❌ 播放失敗: {e}")
            return False