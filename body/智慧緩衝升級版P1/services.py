# services.py
import os
import wave
import numpy as np
import torch
import torchaudio

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

    def transcribe(self, audio_np):
        """語音轉文字 (Whisper)"""
        if audio_np is None: return ""
        try:
            audio_f32 = audio_np.astype(np.float32) / 32768.0
            segments, _ = self.models.stt_model.transcribe(audio_f32, beam_size=5)
            return "".join(seg.text for s in segments for seg in [s]).strip()
        except Exception as e:
            print(f"❌ STT 失敗: {e}")
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

            # 中文模型的標籤映射
            if language == "zh":
                label = config.EMOTION_ZH_LABEL_MAP.get(label, label)

            if top['score'] < config.EMOTION_CONFIDENCE_THRESHOLD['text']: return None
            return {"emotion": label, "score": top['score']}
        except:
            return None

    def detect_audio_emotion(self, audio_path: str, language: str) -> dict:
        """偵測語音情緒 (適配 Emotion2Vec+ Large)"""
        # 檢查模型是否存在 (注意：變數名稱雖叫 pipeline，但內容物現在是 FunASR model)
        if not config.ENABLE_EMOTION_DETECTION or not getattr(self.models, 'emotion_audio_pipeline', None):
            return None

        model = self.models.emotion_audio_pipeline

        try:
            # [修改] 使用 FunASR 推論接口
            # granularity="utterance" 代表分析整句語音
            res = model.generate(
                input=audio_path,
                granularity="utterance",
                extract_embedding=False  # <--- 關鍵參數
            )
            if not res: return None

            # 解析結果
            # Emotion2Vec+ 輸出範例: [{'key': 'wav', 'text': 'happy', 'scores': [0.01, 0.02, ...]}]
            result_data = res[0]
            scores_list = result_data.get('scores', [])

            # Emotion2Vec+ 固定的標籤順序 (依照官方定義)
            labels_order = ["angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised", "unknown"]

            mapped_results = []

            # 1. 將原始分數與標籤配對
            if len(scores_list) == len(labels_order):
                for i, score in enumerate(scores_list):
                    raw_label = labels_order[i]
                    # 映射到系統內部標籤 (例如 happy -> happiness)
                    system_label = config.EMOTION_AUDIO_LABEL_MAP.get(raw_label, raw_label)
                    mapped_results.append({'label': system_label, 'score': float(score)})
            else:
                # 防呆：萬一模型只回傳 text (無 scores)
                top_text = result_data.get('text', 'neutral')
                system_label = config.EMOTION_AUDIO_LABEL_MAP.get(top_text, top_text)
                mapped_results.append({'label': system_label, 'score': 0.99})

            # 2. 應用您的加權系統 (保留 config.py 中的權重邏輯)
            adjusted_results = []
            for item in mapped_results:
                lbl = item['label']
                sc = item['score']

                # 取得權重 (預設為 1.0)
                weight = config.EMOTION_AUDIO_WEIGHTS.get(lbl, 1.0)
                adjusted_score = sc * weight

                adjusted_results.append({
                    'label': lbl,
                    'score': adjusted_score,
                    'original_score': sc,
                    'weight': weight
                })

            # 3. 找出加權後最高分的情緒
            top_result = max(adjusted_results, key=lambda x: x['score'])

            # 4. 檢查信心度閾值
            if top_result['score'] < config.EMOTION_CONFIDENCE_THRESHOLD['audio']:
                # print(f"   ⚠️  語音情緒信心不足: {top_result['score']:.2%}") # Debug用
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

    def select_personality_auto(self, text_emotion: str, audio_emotion: str, facial_emotion: str,
                                emotion_memory: utils.EmotionMemory,
                                text_conf: float = 0, audio_conf: float = 0, facial_conf: float = 0) -> str:
        """根據多模態情緒選擇人格"""
        scores = {}

        def add(e, c, w):
            if e:
                n = config.EMOTION_NORMALIZATION.get(e, e)
                scores[n] = scores.get(n, 0) + c * config.MODALITY_WEIGHTS[w]

        add(text_emotion, text_conf, "text")
        add(audio_emotion, audio_conf, "audio")
        add(facial_emotion, facial_conf, "facial")

        if not scores:
            return config.EMOTION_TO_PERSONALITY.get(emotion_memory.get_dominant_emotion(), "humorous")

        primary = max(scores.items(), key=lambda x: x[1])[0]

        # ====== 🚨 修改區塊：情緒趨勢干預檢查 (新增 print 輸出) ======
        emotion_trend = emotion_memory.get_emotion_trend()

        if emotion_trend == "worsening":
            print("   🚨 記憶趨勢偵測: 情緒【持續惡化】，強制切換至【安撫型】人格 (comforting)。")
            return "comforting"
        elif emotion_trend == "improving":
            print("   👍 記憶趨勢偵測: 情緒正在【改善】。")
        else:
            print("   ℹ️ 記憶趨勢偵測: 情緒狀態【穩定】。")
        # ==========================================================
        personality = config.EMOTION_TO_PERSONALITY.get(primary, "humorous")

        # 中性情緒時隨機切換
        if primary == "neutral":
            import random
            personality = random.choice(["humorous", "rational", "cheerful"])
        return personality

    def generate_response(self, user_text: str, personality: str, conversation_history: list,
                          text_emotion: dict = None, audio_emotion: dict = None,
                          facial_emotion: dict = None) -> (str, list):
        """(LLM) 產生回應"""
        p_config = config.PERSONALITY_CONFIGS[personality]
        language = utils.detect_language(user_text)

        # System Prompt
        system_prompt = p_config["prompt_prefix_zh"] if language == "zh" else p_config["prompt_prefix_en"]

        # 加入情緒狀態
        if text_emotion or audio_emotion or facial_emotion:
            desc = "\n\n【狀態】" if language == "zh" else "\n\n【User State】"
            if text_emotion: desc += f" Text:{text_emotion['emotion']}"
            if audio_emotion: desc += f" Audio:{audio_emotion['emotion']}"
            if facial_emotion: desc += f" Face:{facial_emotion['emotion']}"
            system_prompt += desc

        messages = [{"role": "system", "content": system_prompt}] + conversation_history[-10:]

        # 將指令附加在使用者輸入後
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

    # services.py - text_to_speech_cosvoice 方法更新 (最終、最穩定版本)

    def text_to_speech_cosvoice(self, text: str, output_path: str, personality: str) -> bool:
        """
        (TTS) CosyVoice 2.0 - 支援動態切換音色
        - 根據 config.COSVOICE_VARIANTS 自動切換不同人格的參考音檔
        - 若無設定則使用預設音檔
        """
        try:
            from cosyvoice.utils.file_utils import load_wav
            from zhconv import convert  # 確保有匯入簡繁轉換

            # =================================================
            # 🆕 步驟 1：根據人格選擇參考音訊與逐字稿
            # =================================================
            current_ref_wav = config.COSVOICE_REFERENCE_WAV
            current_ref_text = getattr(config, 'COSVOICE_REFERENCE_TEXT', "")

            # 檢查 config 是否有多重語音設定
            if hasattr(config, 'COSVOICE_VARIANTS') and personality in config.COSVOICE_VARIANTS:
                variant = config.COSVOICE_VARIANTS[personality]
                # 取得該人格專屬的 WAV 和 文字
                cand_wav = variant.get("wav")
                cand_text = variant.get("text", "")

                # 確認檔案真的存在
                if cand_wav and os.path.exists(cand_wav):
                    current_ref_wav = cand_wav
                    current_ref_text = cand_text
                    print(f"   🔊 切換音色: 使用 [{personality}] 專屬語音")
                else:
                    print(f"   ⚠️ 找不到 [{personality}] 的音檔 ({cand_wav})，使用預設")
            else:
                print(f"   🔊 使用預設音色 (Fallback)")

            # 最後防呆檢查：如果選定的檔案不存在，強制回退到預設
            if not os.path.exists(current_ref_wav):
                print(f"❌ 嚴重錯誤：找不到參考音訊 {current_ref_wav}，嘗試使用預設...")
                current_ref_wav = config.COSVOICE_REFERENCE_WAV
                if not os.path.exists(current_ref_wav):
                    print("❌ 連預設音訊都找不到，無法合成")
                    return False

            # =================================================
            # 步驟 2：載入與處理參考音訊
            # =================================================
            prompt_speech_16k = load_wav(str(current_ref_wav), 16000)

            # 準備參考文本 (必須轉簡體以符合模型訓練資料)
            prompt_text = torch.zeros(1, 0, dtype=torch.int32)
            if current_ref_text:
                prompt_text = convert(current_ref_text, 'zh-cn')

            # =================================================
            # 步驟 3：處理目標文字 (Target Text)
            # =================================================
            # 簡單的清理：將換行轉句號，移除星號
            clean_text = text.replace("- ", "，").replace("*", "").replace("\n", "。")
            target_lang = utils.detect_language(text)

            # 格式化輸入
            if target_lang == "zh":
                # 轉簡體是為了提高模型發音準確性
                final_input_text = convert(clean_text, 'zh-cn')
            else:
                final_input_text = clean_text

            utils.debug_log(f"CosyVoice 合成輸入: {final_input_text[:40]}... (人格: {personality})", "DEBUG")

            # =================================================
            # 步驟 4：執行推論 (Inference)
            # =================================================
            output_gen = self.models.cosvoice_model.inference_zero_shot(
                final_input_text, prompt_text, prompt_speech_16k, stream=True
            )

            # =================================================
            # 步驟 5：儲存結果
            # =================================================
            segments = [res['tts_speech'] for res in output_gen]
            if not segments:
                raise Exception("無音訊生成")

            final_audio = torch.cat(segments, dim=1).cpu()

            # 轉為 int16 格式
            audio_int16 = (final_audio * 32767).clamp(-32768, 32767).to(torch.int16)

            # 存檔 (Sample Rate 22050 是 CosyVoice 的預設)
            torchaudio.save(str(output_path), audio_int16, 22050)

            print(f"✅ 語音合成完成 ({personality})")
            return True

        except Exception as e:
            print(f"❌ CosyVoice 失敗: {e}")
            import traceback
            traceback.print_exc()
            return False