# main.py
import os # <--- 確保這行已經存在或新增

import time
import threading
import logging
from scipy.io.wavfile import write
# from scipy.io.wavfile import write # [移除] scipy.io.wavfile，改用 torchaudio
import torch                         # [新增]
import torchaudio                    # [新增]
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

        while True:
            try:
                turn_count += 1
                turn_data = {"turn": turn_count}

                # ===== 階段 1: 等待喚醒 =====
                print(f"👂 正在監聽喚醒詞「{config.WAKE_WORD}」...")
                wake_audio = utils.record_until_silence(
                    self.models.vad,
                    config.SILENCE_TIMEOUT_WAKE,
                    self.models.audio_device_index,
                    sync_face_capture=False
                )

                wake_text = self.services.transcribe(wake_audio)
                print(f"🗣️  喚醒辨識: {wake_text}")
                if not utils.has_wake_word(wake_text):
                    continue

                print("✅ 喚醒成功！請說話...")
                wake_language = utils.detect_language(wake_text)
                wake_confirm_file = config.WAKE_CONFIRM_FILE_ZH if wake_language == "zh" else config.WAKE_CONFIRM_FILE_EN
                if wake_confirm_file.exists():
                    utils.play_audio(str(wake_confirm_file))

                # ===== 階段 2: 主講內容 (同步) =====
                stop_event = threading.Event()
                captured_image_paths = []
                face_thread = None

                if self.models.facial_detector_enabled:
                    face_thread = threading.Thread(
                        target=utils.face_capture_worker,
                        args=(stop_event, captured_image_paths, config.CAMERA_INDEX)
                    )
                    face_thread.start()

                main_audio = utils.record_until_silence(
                    self.models.vad,
                    config.SILENCE_TIMEOUT_MAIN,
                    self.models.audio_device_index,
                    sync_face_capture=True
                )

                if self.models.facial_detector_enabled:
                    stop_event.set()
                    face_thread.join()

                if main_audio is None:
                    print("⚠️  沒收到內容，回待命\n");
                    continue

                # ===== 階段 3: 轉錄與指令檢查 (新增音訊增益標準化) =====

                # [新增] 音訊增益標準化處理 (避免模型因音量過低而只判斷為中性)
                peak = np.abs(main_audio).max()  # np.abs 和 np.max 來自 main_audio 的 numpy array
                if peak > 0 and peak < 32767:
                    scale_factor = 26000 / peak
                    main_audio = (main_audio * scale_factor).astype(np.int16)
                    utils.debug_log(f"🎤 增益調整，因子: {scale_factor:.2f}", "INFO")

                # [修正] 使用 torchaudio 寫入標準 WAV 檔頭 (需確保 torch, torchaudio 已匯入)
                # 1. 將 numpy array (int16) 轉換為 Tensor (float32)
                audio_tensor = torch.from_numpy(main_audio).float().unsqueeze(0)
                # 2. 將 PCM 範圍歸一化到 -1.0 到 1.0 範圍
                audio_tensor = audio_tensor / 32768.0
                # 3. 儲存
                torchaudio.save(str(config.USER_AUDIO_WAV), audio_tensor, config.SAMPLE_RATE)

                user_text = self.services.transcribe(main_audio)
                print(f"\n💬 您說: {user_text}")
                if not user_text:
                    print("⚠️  無法辨識內容，回待命\n");
                    continue

                language = utils.detect_language(user_text)
                is_command, target = utils.check_personality_switch_command(user_text, language)

                if is_command:
                    if target == 'reset':
                        self.manual_personality = None  # 更新狀態
                        reply = "好的！我已經切回自動模式了。" if language == "zh" else "Okay! Switched back to auto mode."
                        print(f"\n🔄 {'已切回自動模式！'}")
                    else:
                        self.manual_personality = target  # 更新狀態
                        cfg = config.PERSONALITY_CONFIGS[target]
                        name = cfg['name'] if language == "zh" else cfg['name_en']
                        print(f"\n✅ 已切換至: {cfg['icon']} {name}")
                        reply = f"好的！我現在切換到{name}了。" if language == "zh" else f"Okay! I've switched to {name} mode."

                    switch_sound = config.PERSONALITY_SWITCH_FILE_ZH if language == "zh" else config.PERSONALITY_SWITCH_FILE_EN
                    if switch_sound.exists():
                        utils.play_audio(str(switch_sound))

                    # 修改：使用 CosyVoice
                    if self.services.text_to_speech_cosvoice(reply, config.REPLY_WAV,
                                                             self.manual_personality or "humorous"):
                        utils.play_audio(config.REPLY_WAV)
                    print(f"\n{'─' * 80}\n🔁 回到待命狀態...\n");
                    continue  # 結束本輪，回到待命

                # ===== 階段 4: 情緒分析(三種模態) =====
                print("\n🔍 進行多模態情緒分析...")
                turn_data["user_text"] = user_text

                text_emotion = self.services.detect_text_emotion(user_text, language)
                turn_data["text_emotion"] = text_emotion['emotion'] if text_emotion else None

                audio_emotion = self.services.detect_audio_emotion(str(config.USER_AUDIO_WAV), language)
                turn_data["audio_emotion"] = audio_emotion['emotion'] if audio_emotion else None

                facial_emotion = self.services.analyze_facial_emotion_from_images(captured_image_paths)
                turn_data["facial_emotion"] = facial_emotion['emotion'] if facial_emotion else None

                if text_emotion or audio_emotion or facial_emotion:
                    print("\n" + utils.format_emotion_display(text_emotion, audio_emotion, facial_emotion, language))

                # ===== 階段 5: 人格選擇 =====
                if self.manual_personality:
                    selected_personality = self.manual_personality
                    turn_data["personality_mode"] = "manual"
                    print(f"\n🎭 目前人格: {config.PERSONALITY_CONFIGS[selected_personality]['icon']} (手動指定)")
                else:
                    print("\n🎭 自動選擇人格模式...")
                    text_emo = text_emotion['emotion'] if text_emotion else None
                    audio_emo = audio_emotion['emotion'] if audio_emotion else None
                    facial_emo = facial_emotion['emotion'] if facial_emotion else None

                    self.memory.add_emotion(text_emo, audio_emo, facial_emo)  # 更新記憶

                    selected_personality = self.services.select_personality_auto(
                        text_emo, audio_emo, facial_emo, self.memory,
                        text_emotion['score'] if text_emotion else 0,
                        audio_emotion['score'] if audio_emotion else 0,
                        facial_emotion['confidence'] if facial_emotion else 0
                    )
                    turn_data["personality_mode"] = "auto"
                    turn_data["final_emotion"] = facial_emo or audio_emo or text_emo
                    print(
                        f"   ✨ 已選擇: {config.PERSONALITY_CONFIGS[selected_personality]['icon']} {config.PERSONALITY_CONFIGS[selected_personality]['name']}")

                turn_data["personality"] = selected_personality

                # ===== 階段 6: 生成回應 =====
                print("\n🤔 思考中...")
                reply, updated_history = self.services.generate_response(
                    user_text, selected_personality, self.conversation_history,
                    text_emotion, audio_emotion, facial_emotion
                )

                self.conversation_history = updated_history  # 更新狀態

                cfg = config.PERSONALITY_CONFIGS[selected_personality]
                print(f"\n{cfg['icon']} {cfg['name']}回應: {reply}")

                turn_data["reply"] = reply
                self.logger.log_turn(turn_data)
                utils.debug_log(f"本輪對話已記錄到日誌", "SUCCESS")

                # ===== 階段 7: TTS & 播放 =====
                # 修改：使用 CosyVoice
                if self.services.text_to_speech_cosvoice(reply, config.REPLY_WAV, selected_personality):
                    utils.play_audio(config.REPLY_WAV)
                print(f"\n{'─' * 80}\n🔁 回到待命狀態...\n")

            except KeyboardInterrupt:
                print("\n🛑 程式結束。")
                print(f"   圖片已儲存於: {config.IMAGE_DIR}")
                if config.ENABLE_CONVERSATION_LOG:
                    summary = self.logger.get_session_summary()
                    print("\n" + "=" * 80 + "\n📊 本次對話統計\n" + "=" * 80)
                    print(f"   對話輪數: {summary.get('total_turns', 0)}")
                    print(f"   持續時間: {summary.get('duration', 0):.1f} 秒")
                    if summary.get('personalities'):
                        print(f"\n   人格分布:")
                        for p, count in summary['personalities'].items():
                            print(
                                f"      • {config.PERSONALITY_CONFIGS[p]['icon']} {config.PERSONALITY_CONFIGS[p]['name']}: {count} 次")
                print("\n   再見！")
                break
            except Exception as e:
                print(f"⚠️  主迴圈發生例外: {e}")
                import traceback;
                traceback.print_exc();
                time.sleep(1)


# ===== 程式進入點 =====
if __name__ == "__main__":
    assistant = MultimodalAssistant()
    assistant.run()