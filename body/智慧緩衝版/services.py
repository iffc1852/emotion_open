# services.py
import queue      # [新增]
import threading  # [新增]
import time       # [新增]
import sounddevice as sd # [新增] (原本 utils 有用，這裡也要用)
import os
import re  # 🚀 [新增] 用於清洗 Emoji 和特殊符號
import torch
import torchaudio
import numpy as np
import config
import utils
from zhconv import convert
from body_emotion_detector import BodyLanguageDetector


class AIServices:
    def __init__(self, models):
        self.models = models
        # 初始化肢體偵測器
        self.body_detector = BodyLanguageDetector()

    def transcribe(self, audio_data):
        """語音轉文字 (Whisper)"""
        if audio_data is None: return ""
        try:
            # 正規化
            audio_float = audio_data.astype(np.float32) / 32768.0

            # 確保變數名稱與 model_loader.py 一致 (stt_model)
            segments, _ = self.models.stt_model.transcribe(
                audio_float,
                beam_size=1 if config.WHISPER_DEVICE == "cpu" else 5,
                language="zh"
            )
            text = "".join([segment.text for segment in segments])
            return text.strip()
        except Exception as e:
            print(f"❌ STT 錯誤: {e}")
            return ""

    def detect_text_emotion(self, text, lang="zh"):
        """文字情緒分析 (修復 LABEL_0 問題)"""
        if not text: return None
        try:
            if hasattr(self.models, 'emotion_text_zh') and self.models.emotion_text_zh:
                results = self.models.emotion_text_zh(text)

                # 防止 HuggingFace 回傳空值或格式錯誤
                if not results: return None

                # 處理可能是 list of list 的情況
                top_result = results[0]
                if isinstance(top_result, list): top_result = top_result[0]

                raw_label = top_result['label']
                score = top_result['score']

                # 🚀 [關鍵修正] 使用 config 對照表將 LABEL_0 轉為 neutral
                final_label = config.EMOTION_ZH_LABEL_MAP.get(raw_label, raw_label)

                return {"emotion": final_label, "score": score}

            return {"emotion": "neutral", "score": 0.5}
        except Exception as e:
            print(f"❌ 文字情緒錯誤: {e}")
            return {"emotion": "neutral", "score": 0.5}

    def detect_audio_emotion(self, audio_path, lang="zh"):
        """語音情緒分析 (Emotion2Vec+)"""
        try:
            # 檢查模型是否存在
            if not hasattr(self.models, 'emotion_audio_pipeline') or self.models.emotion_audio_pipeline is None:
                return None

            # 使用 .generate() 方法傳入 input 參數
            rec_result = self.models.emotion_audio_pipeline.generate(
                input=audio_path,
                granularity="utterance",
                extract_embedding=False
            )

            if rec_result and len(rec_result) > 0:
                # 解析 FunASR 輸出
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

    def analyze_facial_emotion_from_images(self, image_paths):
        """人臉情緒分析 (Py-FEAT)"""
        # 轉發給 facial_emotion_detector_pyfeat.py 的全域函數
        from facial_emotion_detector_pyfeat import analyze_images
        return analyze_images(image_paths)

    def detect_body_emotion(self, image_paths):
        """肢體情緒分析"""
        try:
            return self.body_detector.analyze_pose(image_paths)
        except Exception as e:
            print(f"❌ 肢體分析失敗: {e}")
            return {"emotion": "neutral", "score": 0.0, "label": "無"}

    def generate_response(self, user_text, personality, history, text_emo, audio_emo, face_emo):
        """生成回應 (LLM) - 含清洗邏輯"""
        try:
            # 1. 建構 Prompt
            # 取得該人格的 Prompt 設定
            system_prompt = config.PERSONALITY_CONFIGS[personality]['prompt_prefix_zh']
            messages = [{"role": "system", "content": system_prompt}]

            # 加入歷史紀錄 (最近 4 輪)
            for h in history[-4:]:
                messages.append(h)

            # 加入使用者這句話
            messages.append({"role": "user", "content": user_text})

            # 2. 呼叫模型 (連接 model_loader 的 client)
            completion = self.models.llm_client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=messages,
                max_tokens=config.LLM_MAX_TOKENS,
                temperature=config.LLM_TEMPERATURE
            )
            reply = completion.choices[0].message.content

            # ---------------------------------------------------------
            # 🚀 [新增] 強制清洗回應 (去除 Emoji、星號、多餘換行)
            # ---------------------------------------------------------
            if reply:
                # 去除常見 Emoji (Unicode 範圍)
                reply = re.sub(r'[\U00010000-\U0010ffff]', '', reply)
                # 去除動作標記和全形括號
                reply = reply.replace("*", "").replace("（", "(").replace("）", ")")
                # 去除前後空白和換行
                reply = reply.strip()

                # 如果清洗完變空了，給個預設值
                if not reply:
                    reply = "嘿嘿，我都不知道該說什麼了！"
            # ---------------------------------------------------------

        except Exception as e:
            print(f"❌ LLM 生成失敗: {e}")
            reply = "抱歉，我現在有點秀逗。"

        # 更新歷史
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": reply})
        return reply, history

    def select_personality_auto(self, text_emo, audio_emo, face_emo, memory,
                                text_score=0, audio_score=0, face_score=0,
                                body_emo=None, body_score=0):  # 🆕 完整包含肢體參數
        """
        自動選擇人格 (四模態投票版)
        """
        try:
            candidates = {}
            # 讀取權重 (預設四模態平均)
            weights = getattr(config, 'MODALITY_WEIGHTS', {"text": 0.25, "audio": 0.25, "facial": 0.25, "body": 0.25})

            def vote(emotion, weight, score):
                if not emotion: return
                # 正規化情緒標籤
                norm_emo = config.EMOTION_NORMALIZATION.get(emotion, emotion)
                # 查表找對應人格
                target_p = config.EMOTION_TO_PERSONALITY.get(norm_emo)

                if target_p:
                    if target_p not in candidates: candidates[target_p] = 0.0
                    candidates[target_p] += weight * score

            # 印出權重以供 Debug
            # print(f"   🗳️ 開始投票 (權重: T={weights.get('text')}, A={weights.get('audio')}, F={weights.get('facial')}, B={weights.get('body')})")

            # 1. 文字投票
            vote(text_emo, weights.get("text", 0.3), text_score)
            # 2. 語音投票
            vote(audio_emo, weights.get("audio", 0.3), audio_score)
            # 3. 人臉投票
            vote(face_emo, weights.get("facial", 0.2), face_score)
            # 4. 🆕 肢體投票
            vote(body_emo, weights.get("body", 0.2), body_score)

            # 記憶加權 (讓情緒有點延續性)
            dominant_prev = memory.get_dominant_emotion()
            if dominant_prev and dominant_prev != "neutral":
                norm_prev = config.EMOTION_NORMALIZATION.get(dominant_prev, dominant_prev)
                target_prev = config.EMOTION_TO_PERSONALITY.get(norm_prev)
                if target_prev and target_prev in candidates:
                    candidates[target_prev] += 0.05

            if not candidates:
                print("   🤔 無明顯情緒特徵，使用預設人格")
                return config.DEFAULT_PERSONALITY if config.DEFAULT_PERSONALITY != "auto" else "humorous"

            # 選出分數最高的人格
            best_personality = max(candidates, key=candidates.get)

            # 顯示詳細戰況
            debug_info = ", ".join([f"{p}: {s:.2f}" for p, s in candidates.items()])
            print(f"   📊 戰況: {{{debug_info}}} -> 勝出: {best_personality}")

            return best_personality

        except Exception as e:
            print(f"⚠️ 人格選擇邏輯錯誤: {e}，回退至預設")
            import traceback
            traceback.print_exc()
            return "rational"

    def text_to_speech_cosvoice(self, text: str, output_path: str, personality: str) -> bool:
        """(TTS) CosyVoice 2.0 - 極速除錯版 (Threshold=1 + 計時)"""
        try:
            from cosyvoice.utils.file_utils import load_wav
            from zhconv import convert
            import sounddevice as sd
            import time
            import queue
            import threading

            t0 = time.time()  # [計時] 開始
            print(f"   ⏱️ [0.00s] TTS 請求啟動...")

            # 1. 準備參考音訊與文本
            current_ref_wav = config.COSVOICE_REFERENCE_WAV
            current_ref_text = getattr(config, 'COSVOICE_REFERENCE_TEXT', "")

            if hasattr(config, 'COSVOICE_VARIANTS') and personality in config.COSVOICE_VARIANTS:
                variant = config.COSVOICE_VARIANTS[personality]
                cand_wav = variant.get("wav")
                cand_text = variant.get("text", "")
                if cand_wav and os.path.exists(cand_wav):
                    current_ref_wav = cand_wav
                    current_ref_text = cand_text
                    # print(f"   🔊 切換音色: [{personality}]")

            if not os.path.exists(current_ref_wav):
                print(f"❌ 找不到參考音訊")
                return False

            prompt_speech_16k = load_wav(str(current_ref_wav), 16000)
            prompt_text = torch.zeros(1, 0, dtype=torch.int32)
            if current_ref_text: prompt_text = convert(current_ref_text, 'zh-cn')

            clean_text = text.replace("- ", "，").replace("*", "").replace("\n", "。")
            final_input_text = convert(clean_text, 'zh-cn') if utils.detect_language(text) == "zh" else clean_text

            t1 = time.time()
            print(f"   ⏱️ [{t1 - t0:.2f}s] 前置處理完成，開始呼叫模型推理...")

            # 2. 建立生成器
            output_gen = self.models.cosvoice_model.inference_zero_shot(
                final_input_text, prompt_text, prompt_speech_16k, stream=True
            )

            # 3. 設定參數
            target_sr = 22050
            buffer_queue = queue.Queue()
            full_audio_segments = []

            # 🔥 極速測試：改成 1 (收到第1個片段就馬上播)
            # 如果這樣還是慢，代表是顯卡生成第1段本身就慢
            START_THRESHOLD = 4

            # 生產者
            def producer():
                try:
                    for i, res in enumerate(output_gen):
                        if i == 0:
                            print(f"   ⏱️ [{time.time() - t0:.2f}s] 🔥 GPU 產出第 1 個片段！")

                        chunk_tensor = res['tts_speech']
                        chunk_numpy = chunk_tensor.squeeze().cpu().numpy()
                        buffer_queue.put(chunk_numpy)
                        full_audio_segments.append(chunk_tensor)
                    buffer_queue.put(None)
                except Exception as e:
                    print(f"❌ 生成錯誤: {e}")
                    buffer_queue.put(None)

            t = threading.Thread(target=producer)
            t.start()

            # 消費者
            accumulated_chunks = []
            started_playing = False

            with sd.OutputStream(samplerate=target_sr, channels=1, dtype='float32') as stream:
                while True:
                    try:
                        timeout = 0.1 if started_playing else 5.0  # 等久一點避免誤判
                        chunk = buffer_queue.get(timeout=timeout)
                    except queue.Empty:
                        if t.is_alive():
                            if started_playing: print("!", end="", flush=True)  # 卡頓警示
                            continue
                        else:
                            break

                    if chunk is None: break

                    if not started_playing:
                        accumulated_chunks.append(chunk)
                        if len(accumulated_chunks) >= START_THRESHOLD or not t.is_alive():
                            t2 = time.time()
                            print(f"   ⏱️ [{t2 - t0:.2f}s] 🔊 緩衝滿足(閾值{START_THRESHOLD})，開始播放！")
                            for c in accumulated_chunks:
                                stream.write(c)
                            started_playing = True
                    else:
                        stream.write(chunk)

            t.join()
            print(f"   ⏱️ [{time.time() - t0:.2f}s] 播放結束。")

            if full_audio_segments:
                final_audio = torch.cat(full_audio_segments, dim=1).cpu()
                audio_int16 = (final_audio * 32767).clamp(-32768, 32767).to(torch.int16)
                torchaudio.save(str(output_path), audio_int16, target_sr)
                return True

            return False

        except Exception as e:
            print(f"❌ 播放失敗: {e}")
            import traceback;
            traceback.print_exc()
            return False