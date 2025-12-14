# services.py
import os
import wave
import numpy as np
import torch
import torchaudio
import traceback

# 嘗試匯入必要套件
try:
    from cosyvoice.utils.file_utils import load_wav
except ImportError:
    pass

# 嘗試匯入繁簡轉換
try:
    from zhconv import convert
except ImportError:
    def convert(t, _):
        return t

import config
import utils
from facial_emotion_detector_pyfeat import analyze_images


class AIServices:
    def __init__(self, models):
        self.models = models

    # 🟢 [保留修復] STT 必須用這個版本，否則會出現 "Thanks for watching"
    def transcribe(self, audio_np):
        """語音轉文字 (Whisper) - 自動偵測中英文 + 提示詞引導"""
        if audio_np is None: return ""
        try:
            # 確保輸入是 float32
            if audio_np.dtype != np.float32:
                audio_f32 = audio_np.astype(np.float32) / 32768.0
            else:
                audio_f32 = audio_np

            print(f"   🎤 Whisper 輸入檢查: Max={np.max(np.abs(audio_f32)):.4f}")

            # 執行 Whisper (加入引導詞以防幻覺)
            segments, info = self.models.stt_model.transcribe(
                audio_f32,
                beam_size=5,
                initial_prompt="Hello, 這是繁體中文與英文的混合對話。",
                condition_on_previous_text=False
            )

            print(f"   🌍 Whisper 偵測語言: {info.language.upper()} (信心度: {info.language_probability:.2%})")
            return "".join(seg.text for s in segments for seg in [s]).strip()
        except Exception as e:
            print(f"❌ STT 失敗: {e}")
            traceback.print_exc()
            return ""

    def detect_text_emotion(self, text: str, language: str) -> dict:
        """文字情緒辨識"""
        if not config.ENABLE_EMOTION_DETECTION: return None
        try:
            model = self.models.emotion_text_zh if language == "zh" else self.models.emotion_text_en
            if not model: return None
            results = model(text)
            top = max(results[0], key=lambda x: x['score'])
            label = top['label']

            if language == "zh":
                label = config.EMOTION_ZH_LABEL_MAP.get(label, label)

            if top['score'] < config.EMOTION_CONFIDENCE_THRESHOLD['text']: return None
            return {"emotion": label, "score": top['score']}
        except:
            return None

    def detect_audio_emotion(self, audio_path: str, language: str) -> dict:
        """偵測語音情緒 (Emotion2Vec+)"""
        if not config.ENABLE_EMOTION_DETECTION or not getattr(self.models, 'emotion_audio_pipeline', None):
            return None

        model = self.models.emotion_audio_pipeline

        try:
            res = model.generate(
                input=audio_path,
                granularity="utterance",
                extract_embedding=False
            )
            if not res: return None

            result_data = res[0]
            scores_list = result_data.get('scores', [])
            labels_order = ["angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised", "unknown"]

            mapped_results = []

            if len(scores_list) == len(labels_order):
                for i, score in enumerate(scores_list):
                    raw_label = labels_order[i]
                    system_label = config.EMOTION_AUDIO_LABEL_MAP.get(raw_label, raw_label)
                    mapped_results.append({'label': system_label, 'score': float(score)})
            else:
                top_text = result_data.get('text', 'neutral')
                system_label = config.EMOTION_AUDIO_LABEL_MAP.get(top_text, top_text)
                mapped_results.append({'label': system_label, 'score': 0.99})

            adjusted_results = []
            for item in mapped_results:
                lbl = item['label']
                sc = item['score']
                weight = config.EMOTION_AUDIO_WEIGHTS.get(lbl, 1.0)
                adjusted_score = sc * weight
                adjusted_results.append({'label': lbl, 'score': adjusted_score, 'original_score': sc, 'weight': weight})

            top_result = max(adjusted_results, key=lambda x: x['score'])

            if top_result['score'] < config.EMOTION_CONFIDENCE_THRESHOLD['audio']:
                return None

            return {
                "emotion": top_result['label'],
                "score": top_result['score'],
                "original_score": top_result['original_score'],
                "weight": top_result['weight']
            }

        except Exception as e:
            print(f"❌ 語音情緒辨識錯誤: {e}")
            return None

    def analyze_facial_emotion_from_images(self, image_paths: list) -> dict:
        """人臉情緒辨識"""
        if not self.models.facial_detector_enabled or not image_paths: return None
        utils.debug_log(f"🎭 分析 {len(image_paths)} 幀臉部影像...", "INFO")
        try:
            return analyze_images(image_paths, config.FACIAL_CONFIDENCE_THRESHOLD, config.FACIAL_EMOTION_WEIGHTS)
        except:
            return None

        # services.py (請只替換 select_personality_auto 方法)

    def select_personality_auto(self, text_emotion: str, audio_emotion: str, facial_emotion: str,
                                emotion_memory: 'utils.EmotionMemory',
                                text_conf: float = 0, audio_conf: float = 0, facial_conf: float = 0) -> tuple[
        str, list, str]:
        """
        回傳: (selected_personality, logs, winner_emotion)
        """
        scores = {}
        logs = []
        header = f"🧮 權重設定 (Text:{config.MODALITY_WEIGHTS['text']}, Audio:{config.MODALITY_WEIGHTS['audio']}, Face:{config.MODALITY_WEIGHTS['facial']})"
        print(f"\n   {header}")
        logs.append(header)

        def add(e, c, w):
            if e:
                n = config.EMOTION_NORMALIZATION.get(e, e)
                weight_val = config.MODALITY_WEIGHTS[w]
                added_score = c * weight_val
                scores[n] = scores.get(n, 0) + added_score
                log_line = f"➕ [{w}] {e} ({c:.0%}) x {weight_val} = +{added_score:.4f} -> {n}"
                print(f"     {log_line}")
                logs.append(log_line)

        add(text_emotion, text_conf, "text")
        add(audio_emotion, audio_conf, "audio")
        add(facial_emotion, facial_conf, "facial")

        # 預設贏家
        winner_emotion = "neutral"

        if not scores:
            msg = "⚠️ 無有效情緒，使用預設。"
            print(f"   {msg}")
            logs.append(msg)
            # 沒分數時，贏家是 neutral
            return config.EMOTION_TO_PERSONALITY.get(emotion_memory.get_dominant_emotion(),
                                                     "humorous"), logs, winner_emotion

        # 找出最高分 (Winner!)
        winner_emotion = max(scores.items(), key=lambda x: x[1])[0]
        result_msg = f"🏆 最高分: {winner_emotion} (總分: {scores[winner_emotion]:.4f})"
        print(f"   {result_msg}")
        logs.append(result_msg)

        # 這裡只負責算分，回傳贏家，人格切換判斷留給 server.py

        personality = config.EMOTION_TO_PERSONALITY.get(winner_emotion, "humorous")
        if winner_emotion == "neutral":
            import random
            personality = random.choice(["humorous", "rational", "cheerful"])

        return personality, logs, winner_emotion

    def generate_response(self, user_text: str, personality: str, conversation_history: list,
                          text_emotion: dict = None, audio_emotion: dict = None,
                          facial_emotion: dict = None) -> (str, list):

        """(LLM) 產生回應"""
        p_config = config.PERSONALITY_CONFIGS[personality]
        language = utils.detect_language(user_text)

        system_prompt = p_config["prompt_prefix_zh"] if language == "zh" else p_config["prompt_prefix_en"]

        if text_emotion or audio_emotion or facial_emotion:
            desc = "\n\n【狀態】" if language == "zh" else "\n\n【User State】"
            if text_emotion: desc += f" Text:{text_emotion['emotion']}"
            if audio_emotion: desc += f" Audio:{audio_emotion['emotion']}"
            if facial_emotion: desc += f" Face:{facial_emotion['emotion']}"
            system_prompt += desc

        messages = [{"role": "system", "content": system_prompt}] + conversation_history[-10:] #最近的 10 輪對話

        final_user_content = user_text
        if language == "en":
            final_user_content += "\n(Please reply in English)"
        elif language == "zh":
            final_user_content += "\n(請務必使用繁體中文回答)"

        messages.append({"role": "user", "content": final_user_content})

        try:
            resp = self.models.llm_client.chat.completions.create(
                model=config.LLM_MODEL, messages=messages,
                max_tokens=config.LLM_MAX_TOKENS, temperature=config.LLM_TEMPERATURE
            )
            reply = resp.choices[0].message.content.strip()

            new_history = list(conversation_history)
            new_history.append({"role": "user", "content": user_text})
            new_history.append({"role": "assistant", "content": reply})
            return reply, new_history

        except Exception as e:
            print(f"❌ LLM 錯誤: {e}")
            return "...", conversation_history

    def text_to_speech_cosvoice(self, text: str, output_path: str, personality: str) -> bool:
        """(TTS) CosyVoice 2.0"""
        try:
            from cosyvoice.utils.file_utils import load_wav

            ref_wav = config.COSVOICE_REFERENCE_WAV
            if not ref_wav.exists():
                print(f"❌ 找不到參考音訊: {ref_wav}")
                return False

            prompt_speech_16k = load_wav(str(ref_wav), 16000)
            prompt_text = getattr(config, 'COSVOICE_REFERENCE_TEXT', "")
            if prompt_text:
                from zhconv import convert
                prompt_text = convert(prompt_text, 'zh-cn')

            clean_text = text.replace("- ", "，").replace("*", "").replace("\n", "。")
            target_lang = utils.detect_language(text)

            if target_lang == "zh":
                from zhconv import convert
                final_input_text = convert(clean_text, 'zh-cn')
            else:
                final_input_text = clean_text

            utils.debug_log(f"CosyVoice 合成輸入 (純文本): {final_input_text[:60]}... (人格: {personality})", "DEBUG")

            output_gen = self.models.cosvoice_model.inference_zero_shot(
                final_input_text, prompt_text, prompt_speech_16k, stream=False
            )

            segments = [res['tts_speech'] for res in output_gen]
            if not segments: raise Exception("無音訊生成")

            final_audio = torch.cat(segments, dim=1).cpu()
            audio_int16 = (final_audio * 32767).clamp(-32768, 32767).to(torch.int16)

            torchaudio.save(str(output_path), audio_int16, 22050)
            print(f"✅ 語音合成完成 ({personality})")
            return True

        except Exception as e:
            print(f"❌ CosyVoice 失敗: {e}")
            import traceback
            traceback.print_exc()
            return False