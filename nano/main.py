# main.py
import os # <--- 確保這行已經存在或新增

import time
import threading
import logging
from scipy.io.wavfile import write
import torch                         # [新增]
#import torchaudio                    # [新增]
import numpy as np

# 從本地模組匯入
import config
import utils
from model_loader import ModelManager
from services import AIServices

# ===== 關閉煩人的第三方日誌 =====
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("multipart").setLevel(logging.WARNING)
# ==============================

class MultimodalAssistant:
    """主應用程式類別"""

    def __init__(self):
        # 1. 初始化設定
        utils.initialize_audio_settings()  # 初始化 sounddevice 和 ffmpeg
        utils.clean_image_dir()  # 清理圖片資料夾

        # 2. 載入模型
        # ModelManager 會處理所有模型的載入
        self.models = ModelManager()

        # 3. 實例化 AI 服務
        # 將載入的模型傳遞給 AIServices
        self.services = AIServices(self.models)

        # 4. 初始化狀態
        self.logger = utils.ConversationLogger(config.LOG_DIR)
        self.memory = utils.EmotionMemory()
        self.conversation_history = []  # 對話歷史紀錄 (狀態)
        self.manual_personality = None  # 手動人格 (狀態)

    def _print_welcome_message(self):
        """顯示歡迎和說明訊息"""
        print(f"💡 說出「{config.WAKE_WORD}」來喚醒我")
        personalities = ', '.join([f"{c['icon']}{c['name']}" for c in config.PERSONALITY_CONFIGS.values()])
        print(f"🎭 可用人格: {personalities}")
        print(
            "\n💬 人格切換方法:\n   ✅ 明確指令切換:\n      • 「切換共鳴人格」、「切換溫柔人格」...\n   ✅ 切回自動模式：\n      • 「切回預設人格」、「自動模式」")
        print("\n   ℹ️  預設為自動模式：根據三種情緒(文字+語音+人臉)智慧選擇人格")

        if self.models.facial_detector_enabled:
            print("\n🎭 人臉情緒辨識已啟用 (0.5FPS，與音訊同步)")

        if config.ENABLE_CONVERSATION_LOG or config.ENABLE_DEBUG_LOG:
            print(f"\n📝 日誌系統已啟用 (日誌目錄: {config.LOG_DIR})")
        print("=" * 80 + "\n")

    def run(self):
        """啟動主迴圈"""
        self._print_welcome_message()
        turn_count = 0
        session_id = f"session_{int(time.time())}" 

        while True:
            try:
                turn_count += 1
                turn_data = {"turn": turn_count}
                
                # ===================================================
                # ===== 階段 1: 本地 VAD 喚醒 (短錄音用於 WWD) =====
                # ===================================================
                print(f"👂 正在監聽喚醒詞「{config.WAKE_WORD}」...")
                
                # 1. 錄製短音訊 (只包含喚醒詞)
                wake_audio = utils.record_until_silence(
                    self.models.vad,
                    config.SILENCE_TIMEOUT_WAKE, # <--- 關鍵：使用短靜音時間 (e.g., 1.0s)
                    self.models.audio_device_index,
                    sync_face_capture=False 
                )

                if wake_audio is None:
                    continue
                
                # 2. 本地 WWD 判斷 (STT By Nano CPU)
                print("🧠 正在進行本地 STT 判斷喚醒詞...")
                user_text_wake = self.services.transcribe(wake_audio)
                
                if not utils.has_wake_word(user_text_wake):
                    print(f"🗣️ 喚醒詞「{config.WAKE_WORD}」未通過本地驗證 (STT: {user_text_wake})，回待命\n")
                    continue
                
                # 喚醒成功，播放提示音
                print("✅ 喚醒成功！請說出您的主講內容...")
                wake_confirm_file = config.WAKE_CONFIRM_FILE_ZH
                if wake_confirm_file.exists():
                    utils.play_audio(str(wake_confirm_file))

                # =======================================================
                # ===== 階段 2: 錄製主講內容 (長錄音用於傳輸) =====
                # =======================================================

                # 3. 錄製主講內容
                main_audio = utils.record_until_silence(
                    self.models.vad,
                    config.SILENCE_TIMEOUT_MAIN, # <--- 關鍵：使用長靜音時間 (e.g., 1.5s)
                    self.models.audio_device_index,
                    sync_face_capture=False 
                )
                
                if main_audio is None:
                    # 如果用戶在聽到提示音後沒有說話，則跳過
                    print("⚠️ 未偵測到主講內容，回待命\n")
                    continue
                
                # 4. 儲存主講音訊為 WAV
                peak = np.abs(main_audio).max()
                if peak > 0 and peak < 32767:
                    scale_factor = 26000 / peak
                    main_audio = (main_audio * scale_factor).astype(np.int16)
                    utils.debug_log(f"🎤 增益調整，因子: {scale_factor:.2f}", "INFO")
                
                # 使用 scipy.io.wavfile.write 寫入主講內容
                write(
                    str(config.USER_AUDIO_WAV), 
                    config.SAMPLE_RATE, 
                    main_audio 
                )

                # ======================================================
                # ===== 階段 3: 傳輸到 PC 伺服器 (LLM/TTS 運算) =====
                # ======================================================
                print("🔗 正在將數據傳輸到 PC 伺服器進行 STT/LLM/TTS 運算...")

                # 5. 傳輸核心請求 (傳送 main_audio)
                server_response = self.services.process_multimodal_on_server(
                    turn_data,
                    str(config.USER_AUDIO_WAV),
                    [], 
                    session_id
                )
                
                # 6. 檢查 PC 伺服器回應
                if 'error' in server_response:
                    print(f"❌ 伺服器運算失敗: {server_response.get('error', '未知網絡錯誤')}");
                    print(f"❌ 網絡連線失敗或伺服器錯誤，請檢查 PC 端！");
                    continue
                
                # 7. 處理成功回應
                reply = server_response.get('reply', '抱歉，我現在有點忙。')
                user_text_full = server_response.get('user_text', '') # PC 伺服器進行的完整 STT
                selected_personality = server_response.get('personality', 'humorous')
                
                print(f"\n💬 您說 (STT By PC): {user_text_full}")
                cfg = config.PERSONALITY_CONFIGS[selected_personality]
                print(f"\n{cfg['icon']} {cfg['name']}回應: {reply}")

                # 8. TTS 播放 
                if config.REPLY_WAV.exists():
                    utils.play_audio(config.REPLY_WAV)
                else:
                    print("❌ TTS 音訊下載失敗或不存在。")
                    
                print(f"\n{'─' * 80}\n🔁 回到待命狀態...\n")

            except Exception as e:
                # 這裡保留了對整個迴圈的 Exception 處理
                print(f"⚠️  主迴圈發生例外: {e}")
                import traceback;
                traceback.print_exc();
                time.sleep(1)
                

                
# main.py (在檔案最底部)

# ===== 程式進入點 =====
if __name__ == "__main__":
    import sys
    
    # 第一次初始化
    assistant = MultimodalAssistant()

    while True:
        try:
            # 運行主迴圈
            assistant.run() 
            
        except KeyboardInterrupt:
            # 允許 Ctrl+C 退出
            print("\n服務被手動終止。")
            break
            
        except Exception as e:
            # 捕獲 assistant.run() 函式中拋出的所有致命例外 (包括音訊 IO 錯誤)
            
            # 1. 輸出錯誤堆棧供除錯
            print(f"\n❌ 致命錯誤，服務將退出並由 systemd 重新啟動: {e}")
            import traceback
            traceback.print_exc()
            
            # 2. 錯誤修復流程 (不進行程式內重啟，讓 systemd 接管)
            print("❌ 偵測到硬體或服務連線中斷，將退出程序。")
            print("✅ systemd 的 Restart=1s 設定將在 1 秒後自動啟動新的 Docker 容器。")
            
            # 這裡不執行 time.sleep(1) 和 assistant = MultimodalAssistant()
            # 讓程序自然退出 (這是最關鍵的一步)
            
            # 確保程序退出，而不是無限循環
            # 使用 sys.exit(1) 或讓程序執行到檔案末尾自然終止
            sys.exit(1)