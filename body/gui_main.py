# gui_main.py
import sys
import os
import cv2
import time
import threading
import numpy as np
import torch
import torchaudio
import concurrent.futures
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QProgressBar, QFrame, QSizePolicy)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QPixmap, QFont, QImage, QColor, QPainter, QPaintEvent

import config
import utils
# 這裡匯入 MultimodalAssistant 是為了初始化模型
from main import MultimodalAssistant


# ==========================================
# 元件：可縮放 Label
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
# 執行緒：攝影機
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
                # GUI 顯示用
                rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                self.change_pixmap_signal.emit(convert_to_Qt_format)

                # AI 分析用 (縮小存檔)
                if self.is_capturing_for_ai:
                    current_time = time.time()
                    if current_time - self.last_save_time >= 2.0:
                        self.last_save_time = current_time
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        save_path = config.IMAGE_DIR / f"frame_{timestamp}.jpg"

                        # 縮小圖片加速分析
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
# 元件：情緒進度條
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
# 核心執行緒：AI Worker (整合肢體偵測)
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
        # 🚀 開啟 4 個執行緒 (文字, 語音, 人臉, 肢體)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    def run(self):
        if not self.assistant:
            self.signal_status.emit("正在載入 AI 模型...")
            self.assistant = MultimodalAssistant()
            self.assistant.models.facial_detector_enabled = True

        self.signal_status.emit(f"等待喚醒詞「{config.WAKE_WORD}」...")

        while self.running:
            try:
                # 1. 監聽
                self.signal_waveform.emit(True)
                wake_audio = utils.record_until_silence(self.assistant.models.vad, config.SILENCE_TIMEOUT_WAKE,
                                                        self.assistant.models.audio_device_index)
                self.signal_waveform.emit(False)

                wake_text = self.assistant.services.transcribe(wake_audio)
                if not utils.has_wake_word(wake_text): continue

                # 2. 喚醒
                self.signal_status.emit("✅ 聽到了！請說話...")
                if config.WAKE_CONFIRM_FILE_ZH.exists(): utils.play_audio(str(config.WAKE_CONFIRM_FILE_ZH))

                # 3. 錄音 + 拍照
                self.signal_waveform.emit(True)
                self.camera_thread.start_capture_analysis()
                main_audio = utils.record_until_silence(self.assistant.models.vad, config.SILENCE_TIMEOUT_MAIN,
                                                        self.assistant.models.audio_device_index,
                                                        sync_face_capture=False)
                captured_paths = self.camera_thread.stop_capture_analysis()
                self.signal_waveform.emit(False)

                if main_audio is None:
                    self.signal_status.emit("沒聽到聲音...")
                    continue

                # 4. 轉錄 & 指令
                user_text = self.assistant.services.transcribe(main_audio)
                self.signal_user_text.emit(user_text)

                # 指令切換邏輯
                language = utils.detect_language(user_text)
                is_command, target = utils.check_personality_switch_command(user_text, language)
                if is_command:
                    switch_sound = config.PERSONALITY_SWITCH_FILE_ZH
                    if switch_sound.exists(): utils.play_audio(str(switch_sound))
                    self.assistant.manual_personality = target if target != 'reset' else None
                    reply = "好的！切換模式。"
                    self.signal_robot_text.emit(reply)
                    self.assistant.services.text_to_speech_cosvoice(reply, config.REPLY_WAV,
                                                                    target if target else "humorous")
                    utils.play_audio(config.REPLY_WAV)
                    continue

                # 5. 平行情緒分析 (4 Thread)
                self.signal_status.emit("正在全模態分析情緒...")

                # 儲存音訊供情緒模型讀取
                audio_tensor = torch.from_numpy(main_audio).float().unsqueeze(0) / 32768.0
                torchaudio.save(str(config.USER_AUDIO_WAV), audio_tensor, config.SAMPLE_RATE)

                future_text = self.executor.submit(self.assistant.services.detect_text_emotion, user_text, "zh")
                future_audio = self.executor.submit(self.assistant.services.detect_audio_emotion,
                                                    str(config.USER_AUDIO_WAV), "zh")
                future_face = self.executor.submit(self.assistant.services.analyze_facial_emotion_from_images,
                                                   captured_paths)
                future_body = self.executor.submit(self.assistant.services.detect_body_emotion, captured_paths)  # 🆕 肢體

                text_emo = future_text.result()
                audio_emo = future_audio.result()
                face_emo = future_face.result()
                body_emo = future_body.result()  # 🆕

                gui_data = {
                    "text_score": text_emo['score'] if text_emo else 0.0,
                    "text_label": config.EMOTION_ZH_MAP.get(text_emo['emotion'],
                                                            text_emo['emotion']) if text_emo else "無",
                    "audio_score": audio_emo['score'] if audio_emo else 0.0,
                    "audio_label": config.EMOTION_ZH_MAP.get(audio_emo['emotion'],
                                                             audio_emo['emotion']) if audio_emo else "無",
                    "face_score": face_emo['confidence'] if face_emo else 0.0,
                    "face_label": face_emo['emotion_zh'] if face_emo else "無",
                    "body_score": body_emo['score'],
                    "body_label": body_emo['label']
                }
                self.signal_emotions.emit(gui_data)

                # --- 階段 6: 生成回應 ---
                self.signal_status.emit("思考中...")

                # 🚀 [修正] 傳入 body_emo 和 body_score
                personality = self.assistant.manual_personality or self.assistant.services.select_personality_auto(
                    text_emo['emotion'] if text_emo else None,
                    audio_emo['emotion'] if audio_emo else None,
                    face_emo['emotion'] if face_emo else None,
                    self.assistant.memory,

                    text_score=text_emo['score'] if text_emo else 0.0,
                    audio_score=audio_emo['score'] if audio_emo else 0.0,
                    face_score=face_emo['confidence'] if face_emo else 0.0,

                    # 🆕 新增這兩行
                    body_emo=body_emo['emotion'] if body_emo else None,
                    body_score=body_emo['score'] if body_emo else 0.0
                )

                reply, updated_history = self.assistant.services.generate_response(
                    user_text, personality, self.assistant.conversation_history,
                    text_emo, audio_emo, face_emo
                )
                self.assistant.conversation_history = updated_history
                self.signal_robot_text.emit(reply)

                # 📝 寫入日誌
                if config.ENABLE_CONVERSATION_LOG:
                    turn_data = {
                        "turn_id": len(self.assistant.conversation_history) // 2,
                        "user_text": user_text, "bot_text": reply, "personality": personality,
                        "emotions": {"text": text_emo, "audio": audio_emo, "face": face_emo, "body": body_emo},
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    self.assistant.logger.log_turn(turn_data)

                # 7. TTS
                self.signal_status.emit("說話中...")
                if self.assistant.services.text_to_speech_cosvoice(reply, config.REPLY_WAV, personality):
                    utils.play_audio(config.REPLY_WAV)

                self.signal_status.emit(f"等待喚醒詞「{config.WAKE_WORD}」...")

            except Exception as e:
                print(f"❌ Worker Error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1)


# ==========================================
# GUI 主視窗
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EMO+ 多模態情緒感知")
        self.resize(1024, 650)
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
        self.status_label = QLabel("啟動中...")
        title_bar.addWidget(title)
        title_bar.addStretch()
        title_bar.addWidget(self.status_label)
        main_layout.addLayout(title_bar)

        content_layout = QHBoxLayout()

        # 左側
        left_panel = QFrame()
        left_panel.setStyleSheet("QFrame { background-color: white; border-radius: 15px; border: 1px solid #E0E0E0; }")
        left_layout = QVBoxLayout(left_panel)

        lbl_cam = QLabel("影像輸入")
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
            "background-color: #F0F0F0; border-radius: 10px; padding: 10px; font-size: 14px;")
        self.user_text_box.setFixedHeight(80)
        left_layout.addWidget(self.user_text_box)

        self.wave_label = QLabel("〰️〰️〰️")
        self.wave_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.wave_label.setStyleSheet("color: #AAA; font-size: 20px;")
        self.wave_label.setFixedHeight(30)
        left_layout.addWidget(self.wave_label)

        # 右側
        right_panel = QFrame()
        right_panel.setStyleSheet("background-color: transparent;")
        right_layout = QVBoxLayout(right_panel)

        # 情緒條面板
        emo_panel = QFrame()
        emo_panel.setStyleSheet("QFrame { background-color: white; border-radius: 15px; border: 1px solid #E0E0E0; }")
        emo_layout = QVBoxLayout(emo_panel)

        lbl_emo = QLabel("模態信心度")
        lbl_emo.setFont(QFont("Microsoft JhengHei", 12, QFont.Weight.Bold))
        emo_layout.addWidget(lbl_emo)

        self.bar_text = EmotionBar("文字", "#FFD700")
        self.bar_audio = EmotionBar("語音", "#87CEFA")
        self.bar_face = EmotionBar("人臉", "#90EE90")
        self.bar_body = EmotionBar("肢體", "#FF69B4")  # 🆕 新增肢體條 (粉紅)

        emo_layout.addWidget(self.bar_text)
        emo_layout.addWidget(self.bar_audio)
        emo_layout.addWidget(self.bar_face)
        emo_layout.addWidget(self.bar_body)
        emo_layout.addStretch()

        # 機器人面板
        bot_panel = QFrame()
        bot_panel.setStyleSheet("background-color: transparent;")
        bot_layout = QVBoxLayout(bot_panel)

        self.bot_img = ScalableLabel()
        img_path = os.path.join("assets", "robot_avatar.png")
        if os.path.exists(img_path):
            self.bot_img.setPixmap(QPixmap(img_path))
        else:
            self.bot_img.setText("Robot")

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
        self.bar_body.set_value(data.get('body_score', 0), data.get('body_label'))

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