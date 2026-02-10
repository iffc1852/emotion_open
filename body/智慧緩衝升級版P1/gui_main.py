# gui_main.py
import sys
import os
import cv2
import time
import threading
import numpy as np
import torch
import torchaudio
import concurrent.futures  # 🚀 [新增] 用於平行處理
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QProgressBar, QFrame, QSizePolicy)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QPixmap, QFont, QImage, QColor, QPainter, QPaintEvent

# 匯入原本的後端模組
import config
import utils
from main import MultimodalAssistant


# ==========================================
# 🆕 自定義元件：可隨視窗縮放的圖片標籤
# ==========================================
class ScalableLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setMinimumSize(1, 1)
        self._pixmap = None

    def setPixmap(self, pixmap):
        self._pixmap = pixmap
        self.update()

    def paintEvent(self, event: QPaintEvent):
        if not self._pixmap:
            super().paintEvent(event)
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        target_size = self.size()
        scaled_pixmap = self._pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        x = (target_size.width() - scaled_pixmap.width()) // 2
        y = (target_size.height() - scaled_pixmap.height()) // 2
        painter.drawPixmap(x, y, scaled_pixmap)


# ==========================================
# 📷 攝影機執行緒 (優化版：存小圖)
# ==========================================
class CameraThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self.running = True
        self.is_capturing_for_ai = False
        self.captured_paths = []
        self.last_save_time = 0

    def run(self):
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        if not cap.isOpened():
            print(f"❌ 無法開啟攝影機 (Index: {config.CAMERA_INDEX})")
            return

        while self.running:
            ret, cv_img = cap.read()
            if ret:
                # 1. GUI 顯示用 (維持原畫質或適度縮放)
                rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                self.change_pixmap_signal.emit(convert_to_Qt_format)

                # 2. 存圖給 AI (🚀 優化：縮小圖片以加速 CPU 分析)
                if self.is_capturing_for_ai:
                    current_time = time.time()
                    if current_time - self.last_save_time >= 2.0:  # 維持 0.5 FPS
                        self.last_save_time = current_time

                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        save_path = config.IMAGE_DIR / f"frame_{timestamp}.jpg"

                        # 🚀 關鍵優化：將圖片縮小到 320x240
                        # 這對 CPU 跑 Py-FEAT 來說，速度會快非常多！
                        small_img = cv2.resize(cv_img, (320, 240), interpolation=cv2.INTER_AREA)

                        cv2.imwrite(str(save_path), small_img)
                        self.captured_paths.append(str(save_path))

            time.sleep(0.03)

        cap.release()

    def start_capture_analysis(self):
        self.captured_paths = []
        self.last_save_time = 0
        self.is_capturing_for_ai = True

    def stop_capture_analysis(self):
        self.is_capturing_for_ai = False
        return self.captured_paths


# ==========================================
# 🎨 自定義 UI 元件：情緒進度條
# ==========================================
class EmotionBar(QWidget):
    def __init__(self, label_text, color="#FFD700", parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 5, 0, 5)

        self.lbl_name = QLabel(label_text)
        self.lbl_name.setFont(QFont("Microsoft JhengHei", 10, QFont.Weight.Bold))
        self.lbl_name.setFixedWidth(110)

        self.pbar = QProgressBar()
        self.pbar.setTextVisible(False)
        self.pbar.setFixedHeight(12)
        self.pbar.setStyleSheet(f"""
            QProgressBar {{ border: none; background-color: #E0E0E0; border-radius: 6px; }}
            QProgressBar::chunk {{ background-color: {color}; border-radius: 6px; }}
        """)

        self.lbl_val = QLabel("0%")
        self.lbl_val.setFont(QFont("Arial", 10))
        self.lbl_val.setFixedWidth(40)
        self.lbl_val.setAlignment(Qt.AlignmentFlag.AlignRight)

        layout.addWidget(self.lbl_name)
        layout.addWidget(self.pbar)
        layout.addWidget(self.lbl_val)

    def set_value(self, value, emotion_name=None):
        val = max(0.0, min(1.0, value))
        self.pbar.setValue(int(val * 100))
        self.lbl_val.setText(f"{int(val * 100)}%")
        if emotion_name:
            prefix = self.lbl_name.text().split(':')[0].strip()
            self.lbl_name.setText(f"{prefix}: {emotion_name}")


# ==========================================
# 🧠 AI 背景核心執行緒 (🚀 優化：並行處理)
# ==========================================
class AIWorker(QThread):
    signal_status = pyqtSignal(str)
    signal_user_text = pyqtSignal(str)
    signal_robot_text = pyqtSignal(str)
    signal_emotions = pyqtSignal(dict)
    signal_waveform = pyqtSignal(bool)

    def __init__(self, camera_thread_ref):
        super().__init__()
        self.running = True
        self.assistant = None
        self.camera_thread = camera_thread_ref
        # 建立執行緒池，用於並行處理分析任務
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)

    def run(self):
        if not self.assistant:
            self.signal_status.emit("正在載入 AI 模型 (請稍候)...")
            self.assistant = MultimodalAssistant()
            self.assistant.models.facial_detector_enabled = True

        self.signal_status.emit(f"等待喚醒詞「{config.WAKE_WORD}」...")

        while self.running:
            try:
                # --- 階段 1: 監聽喚醒 ---
                self.signal_waveform.emit(True)
                wake_audio = utils.record_until_silence(self.assistant.models.vad, config.SILENCE_TIMEOUT_WAKE,
                                                        self.assistant.models.audio_device_index)
                self.signal_waveform.emit(False)

                wake_text = self.assistant.services.transcribe(wake_audio)
                if not utils.has_wake_word(wake_text): continue

                # --- 階段 2: 喚醒成功 ---
                self.signal_status.emit("✅ 聽到了！請說話...")
                if config.WAKE_CONFIRM_FILE_ZH.exists(): utils.play_audio(str(config.WAKE_CONFIRM_FILE_ZH))

                # --- 階段 3: 主錄音 + 視覺採集 ---
                self.signal_waveform.emit(True)
                self.camera_thread.start_capture_analysis()
                main_audio = utils.record_until_silence(self.assistant.models.vad, config.SILENCE_TIMEOUT_MAIN,
                                                        self.assistant.models.audio_device_index,
                                                        sync_face_capture=False)
                captured_paths = self.camera_thread.stop_capture_analysis()
                self.signal_waveform.emit(False)

                if main_audio is None:
                    self.signal_status.emit("沒聽到聲音，回到待機...")
                    continue

                # --- 階段 4: 轉錄 ---
                user_text = self.assistant.services.transcribe(main_audio)
                self.signal_user_text.emit(user_text)

                # 指令檢查
                language = utils.detect_language(user_text)
                is_command, target = utils.check_personality_switch_command(user_text, language)

                if is_command:
                    switch_sound = config.PERSONALITY_SWITCH_FILE_ZH
                    if switch_sound.exists(): utils.play_audio(str(switch_sound))

                    if target == 'reset':
                        self.assistant.manual_personality = None
                        reply = "好的！已切回自動模式。"
                    else:
                        self.assistant.manual_personality = target
                        cfg = config.PERSONALITY_CONFIGS[target]
                        reply = f"好的！我變身為{cfg['name']}了。"

                    self.signal_robot_text.emit(reply)

                    current_p = target if target != 'reset' else "humorous"
                    if self.assistant.services.text_to_speech_cosvoice(reply, config.REPLY_WAV, current_p):
                        utils.play_audio(config.REPLY_WAV)

                    self.signal_status.emit(f"等待喚醒詞「{config.WAKE_WORD}」...")
                    continue

                    # --- 階段 5: 平行化多模態情緒分析 (🚀 速度優化核心) ---
                self.signal_status.emit("正在分析情緒 (平行處理中)...")
                audio_tensor = torch.from_numpy(main_audio).float().unsqueeze(0) / 32768.0
                torchaudio.save(str(config.USER_AUDIO_WAV), audio_tensor, config.SAMPLE_RATE)

                # 使用 ThreadPool 平行執行三個分析任務
                # 這樣就算 Py-FEAT (CPU) 跑得慢，也不會卡住其他事情，且整體時間 = 最慢的那個任務時間
                future_text = self.executor.submit(self.assistant.services.detect_text_emotion, user_text, "zh")
                future_audio = self.executor.submit(self.assistant.services.detect_audio_emotion,
                                                    str(config.USER_AUDIO_WAV), "zh")
                future_face = self.executor.submit(self.assistant.services.analyze_facial_emotion_from_images,
                                                   captured_paths)

                # 等待結果
                text_emo_data = future_text.result()
                audio_emo_data = future_audio.result()
                face_emo_data = future_face.result()

                gui_data = {
                    "text_score": text_emo_data['score'] if text_emo_data else 0.0,
                    "text_label": config.EMOTION_ZH_MAP.get(text_emo_data['emotion'],
                                                            text_emo_data['emotion']) if text_emo_data else "無",
                    "audio_score": audio_emo_data['score'] if audio_emo_data else 0.0,
                    "audio_label": config.EMOTION_ZH_MAP.get(audio_emo_data['emotion'],
                                                             audio_emo_data['emotion']) if audio_emo_data else "無",
                    "face_score": face_emo_data['confidence'] if face_emo_data else 0.0,
                    "face_label": face_emo_data['emotion_zh'] if face_emo_data else "無"
                }
                self.signal_emotions.emit(gui_data)

                # --- 階段 6: 生成回應 ---
                self.signal_status.emit("思考回應中...")
                if self.assistant.manual_personality:
                    personality = self.assistant.manual_personality
                else:
                    personality = self.assistant.services.select_personality_auto(
                        text_emo_data['emotion'] if text_emo_data else None,
                        audio_emo_data['emotion'] if audio_emo_data else None,
                        face_emo_data['emotion'] if face_emo_data else None,
                        self.assistant.memory
                    )

                reply, updated_history = self.assistant.services.generate_response(
                    user_text, personality, self.assistant.conversation_history,
                    text_emo_data, audio_emo_data, face_emo_data
                )

                self.assistant.conversation_history = updated_history
                self.signal_robot_text.emit(reply)
                # =========== 📝 新增：寫入對話日誌 ===========
                if config.ENABLE_CONVERSATION_LOG:
                    # 整理要記錄的數據
                    turn_data = {
                        "turn_id": len(self.assistant.conversation_history) // 2,
                        "user_text": user_text,
                        "bot_text": reply,
                        "personality": personality,
                        "emotions": {
                            "text": text_emo_data,
                            "audio": audio_emo_data,
                            "face": face_emo_data
                        },
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    # 呼叫 logger 寫入檔案
                    self.assistant.logger.log_turn(turn_data)
                    print(f"📝 已寫入日誌")
                # ===========================================

                # --- 階段 7: TTS ---
                self.signal_status.emit("正在說話...")
                if self.assistant.services.text_to_speech_cosvoice(reply, config.REPLY_WAV, personality):
                    utils.play_audio(config.REPLY_WAV)

                self.signal_status.emit(f"等待喚醒詞「{config.WAKE_WORD}」...")
            except Exception as e:
                print(f"❌ AI 執行緒錯誤: {e}")
                time.sleep(1)


# ==========================================
# 🖥️ 主視窗 GUI
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EMO+ 多模態情緒感知")
        self.resize(1024, 600)
        self.setStyleSheet("background-color: #F5F5F7;")

        self.camera_thread = CameraThread()
        self.camera_thread.change_pixmap_signal.connect(self.update_camera_image)
        self.camera_thread.start()

        self.worker = AIWorker(self.camera_thread)
        self.worker.signal_status.connect(self.update_status)
        self.worker.signal_user_text.connect(self.update_user_text)
        self.worker.signal_robot_text.connect(self.update_robot_text)
        self.worker.signal_emotions.connect(self.update_emotions)
        self.worker.signal_waveform.connect(self.update_waveform)
        self.worker.start()

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # 標題
        title_bar = QHBoxLayout()
        title = QLabel("EMO+ 多模態情緒感知")
        title.setFont(QFont("Microsoft JhengHei", 20, QFont.Weight.Bold))
        title.setStyleSheet("color: #333;")
        self.status_label = QLabel("啟動中...")
        self.status_label.setStyleSheet("color: #666; font-size: 16px;")
        title_bar.addWidget(title)
        title_bar.addStretch()
        title_bar.addWidget(self.status_label)
        main_layout.addLayout(title_bar)

        content_layout = QHBoxLayout()

        # --- 左側 ---
        left_panel = QFrame()
        left_panel.setStyleSheet("QFrame { background-color: white; border-radius: 15px; border: 1px solid #E0E0E0; }")
        left_layout = QVBoxLayout(left_panel)

        lbl_cam = QLabel("語音輸入 (即時影像)")
        lbl_cam.setFont(QFont("Microsoft JhengHei", 12, QFont.Weight.Bold))
        lbl_cam.setFixedHeight(30)
        left_layout.addWidget(lbl_cam)

        self.camera_label = ScalableLabel()
        self.camera_label.setStyleSheet("background-color: black; border-radius: 10px;")
        left_layout.addWidget(self.camera_label, 1)

        self.user_text_box = QLabel("等待輸入...")
        self.user_text_box.setWordWrap(True)
        self.user_text_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.user_text_box.setStyleSheet(
            "background-color: #F0F0F0; border-radius: 10px; padding: 10px; font-size: 14px; color: #333;")
        self.user_text_box.setFixedHeight(80)
        left_layout.addWidget(self.user_text_box)

        self.wave_label = QLabel("〰️〰️〰️")
        self.wave_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.wave_label.setStyleSheet("color: #AAA; font-size: 20px;")
        self.wave_label.setFixedHeight(30)
        left_layout.addWidget(self.wave_label)

        # --- 右側 ---
        right_panel = QFrame()
        right_panel.setStyleSheet("background-color: transparent;")
        right_layout = QVBoxLayout(right_panel)

        # 情緒條
        emo_panel = QFrame()
        emo_panel.setStyleSheet("QFrame { background-color: white; border-radius: 15px; border: 1px solid #E0E0E0; }")
        emo_layout = QVBoxLayout(emo_panel)

        lbl_emo = QLabel("模態情緒信心度 (Confidence)")
        lbl_emo.setFont(QFont("Microsoft JhengHei", 12, QFont.Weight.Bold))
        emo_layout.addWidget(lbl_emo)

        self.bar_text = EmotionBar("文字", "#FFD700")
        self.bar_audio = EmotionBar("語音", "#87CEFA")
        self.bar_face = EmotionBar("人臉", "#90EE90")

        emo_layout.addWidget(self.bar_text)
        emo_layout.addWidget(self.bar_audio)
        emo_layout.addWidget(self.bar_face)
        emo_layout.addStretch()

        # 機器人
        bot_panel = QFrame()
        bot_panel.setStyleSheet("background-color: transparent;")
        bot_layout = QVBoxLayout(bot_panel)

        self.bot_img = ScalableLabel()
        img_path = os.path.join("assets", "robot_avatar.png")
        if os.path.exists(img_path):
            self.bot_img.setPixmap(QPixmap(img_path))
            self.bot_img.setStyleSheet("background-color: transparent;")
        else:
            self.bot_img.setText("Robot")
            self.bot_img.setStyleSheet("background-color: #EEE; border-radius: 75px;")

        bot_layout.addWidget(self.bot_img, 1)

        self.bot_text = QLabel("...")
        self.bot_text.setWordWrap(True)
        self.bot_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.bot_text.setStyleSheet("font-size: 16px; color: #555; font-weight: bold;")
        self.bot_text.setFixedHeight(80)
        bot_layout.addWidget(self.bot_text)

        right_layout.addWidget(emo_panel, 1)
        right_layout.addWidget(bot_panel, 1)

        content_layout.addWidget(left_panel, 1)
        content_layout.addWidget(right_panel, 1)
        main_layout.addLayout(content_layout)

    @pyqtSlot(QImage)
    def update_camera_image(self, qt_image):
        self.camera_label.setPixmap(QPixmap.fromImage(qt_image))

    def update_status(self, text):
        self.status_label.setText(text)

    def update_user_text(self, text):
        self.user_text_box.setText(text)

    def update_robot_text(self, text):
        self.bot_text.setText(text)

    def update_emotions(self, data):
        self.bar_text.set_value(data.get('text_score', 0), data.get('text_label'))
        self.bar_audio.set_value(data.get('audio_score', 0), data.get('audio_label'))
        self.bar_face.set_value(data.get('face_score', 0), data.get('face_label'))

    def update_waveform(self, is_active):
        if is_active:
            self.wave_label.setText("🎙️ 〰️〰️〰️")
            self.wave_label.setStyleSheet("color: #FF5722; font-size: 20px; font-weight: bold;")
        else:
            self.wave_label.setText("〰️〰️〰️")
            self.wave_label.setStyleSheet("color: #AAA; font-size: 20px;")

    def closeEvent(self, event):
        self.camera_thread.running = False
        self.worker.running = False
        # 關閉執行緒池
        self.worker.executor.shutdown(wait=False)
        self.camera_thread.wait()
        self.worker.wait()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    font = QFont("Microsoft JhengHei", 10)
    app.setFont(font)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())