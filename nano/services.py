# services.py (Client 極簡版 - 專注於網絡傳輸)
import os
import requests
import config
import utils
import numpy as np 
import traceback
import json

class AIServices:
    def __init__(self, models):
        self.models = models

# services.py (替換 transcribe 函式)
    def transcribe(self, audio_np):
        """語音轉文字 (OpenAI-Whisper - 本地 WWD 專用)"""
        if audio_np is None: return ""
        try:
            # 1. 格式轉換：確保是 float32 且在 -1~1 之間 
            if audio_np.dtype != np.float32:
                audio_f32 = audio_np.astype(np.float32) / 32768.0
            else:
                audio_f32 = audio_np

            # 2. 執行辨識 (CPU 運算)
            result = self.models.stt_model.transcribe(
                audio_f32,
                beam_size=5,
                initial_prompt="Hello, 這是繁體中文與英文的混合對話。",
                fp16=False, # 強制使用 CPU/FP32，確保穩定
                language=None 
            )

            text = result["text"].strip()
            detected_lang = result.get('language', 'unknown')
            print(f"   🌍 Whisper 偵測語言: {detected_lang}")

            return text

        except Exception as e:
            print(f"❌ 本地 STT 失敗: {e}")
            # traceback.print_exc() # 避免刷屏
            return ""

# (services.py 中的 process_multimodal_on_server 保持不變，因為 PC 仍需要音訊檔)

    def process_multimodal_on_server(self, turn_data, audio_path, image_paths, session_id):
        """將所有數據傳輸到 PC 伺服器進行處理"""
        
        # 1. 準備數據
        files = {'audio_file': open(audio_path, 'rb')}
        for i, img_path in enumerate(image_paths):
            files[f'image_{i}'] = open(img_path, 'rb')
            
        # 2. 準備表單數據
        turn_data['session_id'] = session_id
        form_data = {'turn_data': json.dumps(turn_data)}

        # 3. 執行 POST 請求
        try:
            # 確保 LLM_BASE_URL (ex: http://<PC_IP>:8000/v1) 是正確的
            # 我們將 /v1 替換為實際的處理端點
            url = config.LLM_BASE_URL.replace("/v1", config.API_PROCESS_ENDPOINT)
            utils.debug_log(f"🔗 正在向 PC 伺服器發送請求: {url}", "INFO")
            
            response = requests.post(
                url, 
                files=files, 
                data=form_data, 
                timeout=300 # 設置較長的超時時間
            )
            response.raise_for_status() # 檢查 HTTP 錯誤
            
            # 4. 處理回傳
            server_response_data = response.json()

            # 5. 下載 TTS 音訊
            tts_filename = server_response_data.get('tts_audio_file')
            if tts_filename:
                tts_url = config.LLM_BASE_URL.replace("/v1", config.API_AUDIO_DOWNLOAD_ENDPOINT) + f"?filename={tts_filename}"
                utils.download_and_save_audio(tts_url, config.REPLY_WAV)

            return server_response_data

        except requests.exceptions.RequestException as e:
            utils.debug_log(f"❌ 連線或請求錯誤: {e}", "ERROR")
            return {"error": f"連線錯誤: {e}"}
        except Exception as e:
            utils.debug_log(f"❌ 處理伺服器回應錯誤: {e}", "ERROR")
            return {"error": f"內部處理錯誤: {e}"}


    # 移除所有在 Client 端執行的模型運算功能
    def detect_text_emotion(self, *args): return None
    def detect_audio_emotion(self, *args): return None
    def analyze_facial_emotion_from_images(self, *args): return None
    def select_personality_auto(self, *args): return None
    def generate_response(self, *args): return None
    def text_to_speech_cosvoice(self, *args): return False