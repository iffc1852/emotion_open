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
from zhconv import convert

import config
import utils
from main import MultimodalAssistant


class DynamicTextLabel(QLabel):
    def __init__(self, text, base_size=12, weight=QFont.Weight.Bold, align=Qt.AlignmentFlag.AlignCenter, parent=None):
        super().__init__(text, parent)
        self.base_size = base_size
        self.weight = weight
        self.setAlignment(align)
        self.setFont(QFont("Microsoft JhengHei", base_size, weight))
        self.setMinimumHeight(20)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        new_size = max(self.base_size, int(min(self.width() * 0.05, self.height() * 0.35)))
        font = self.font()
        font.setPointSize(new_size)
        self.setFont(font)


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
# 🚀 [全新] OpenCV 影片播放執行緒 (極度穩定，不會崩潰)
# ==========================================
class AvatarPlayerThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self.running = False
        self.video_path = ""
        self.cap = None

    def play_video(self, path):
        self.video_path = path
        self.running = True
        self.start()

    def stop_video(self):
        self.running = False
        self.wait()

    def run(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print(f"❌ 無法讀取影片: {self.video_path}")
            return

        # 取得影片 FPS 以控制播放速度
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps): fps = 24.0
        delay = 1.0 / fps

        while self.running:
            start_time = time.time()
            ret, frame = self.cap.read()

            if not ret:
                # 影片結束，從頭開始播放 (無限循環)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # 轉換顏色 BGR -> RGB
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w

            # 🚀 務必使用 .copy()，防止記憶體崩潰
            qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
            self.change_pixmap_signal.emit(qt_img)

            # 控制幀率，讓播放速度正常
            elapsed = time.time() - start_time
            sleep_time = delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        self.cap.release()


# ==========================================
class CameraThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self.running = True
        self.is_capturing_for_ai = False
        self.captured_frames = []
        self.last_save_time = 0

    def run(self):
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        if not cap.isOpened():
            print(f"❌ 無法開啟攝影機")
            return

        while self.running:
            ret, cv_img = cap.read()
            if ret:
                rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                self.change_pixmap_signal.emit(convert_to_Qt_format)

                if self.is_capturing_for_ai:
                    current_time = time.time()
                    if current_time - self.last_save_time >= 0.5:
                        self.last_save_time = current_time
                        small_img = cv2.resize(cv_img, (320, 240), interpolation=cv2.INTER_AREA)
                        rgb_small = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)
                        self.captured_frames.append(rgb_small)

            time.sleep(0.03)
        cap.release()

    def start_capture_analysis(self):
        self.captured_frames = []
        self.last_save_time = 0
        self.is_capturing_for_ai = True

    def stop_capture_analysis(self):
        self.is_capturing_for_ai = False
        return self.captured_frames


class EmotionBar(QWidget):
    def __init__(self, label_text, color="#FFD700", parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 5, 0, 5)

        self.lbl_name = DynamicTextLabel(label_text, base_size=10,
                                         align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self.pbar = QProgressBar()
        self.pbar.setTextVisible(False)
        self.pbar.setMinimumHeight(12)
        self.pbar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.pbar.setStyleSheet(f"""
            QProgressBar {{ border: none; background-color: #E0E0E0; border-radius: 6px; }}
            QProgressBar::chunk {{ background-color: {color}; border-radius: 6px; }}
        """)

        self.lbl_val = DynamicTextLabel("0%", base_size=10,
                                        align=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        layout.addWidget(self.lbl_name, 3)
        layout.addWidget(self.pbar, 6)
        layout.addWidget(self.lbl_val, 2)

    def set_value(self, value, emotion_name=None):
        val = max(0.0, min(1.0, value))
        self.pbar.setValue(int(val * 100))
        self.lbl_val.setText(f"{int(val * 100)}%")
        if emotion_name:
            emotion_name_tw = convert(emotion_name, 'zh-tw')
            prefix = self.lbl_name.text().split(':')[0].strip()
            self.lbl_name.setText(f"{prefix}: {emotion_name_tw}")


class AIWorker(QThread):
    signal_status = pyqtSignal(str)
    signal_user_text = pyqtSignal(str)
    signal_robot_text = pyqtSignal(str)
    signal_emotions = pyqtSignal(dict)
    signal_waveform = pyqtSignal(bool)
    signal_speaking = pyqtSignal(bool, str)
    signal_final_result = pyqtSignal(str)

    def __init__(self, camera_thread_ref):
        super().__init__()
        self.running = True
        self.assistant = None
        self.camera_thread = camera_thread_ref
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    def run(self):
        if not self.assistant:
            self.signal_status.emit("正在載入 AI 模型...")
            self.assistant = MultimodalAssistant()
            self.assistant.models.facial_detector_enabled = True
            self.signal_status.emit("正在預熱 GPU 模型...")
            self.assistant.services.warm_up()

        self.signal_status.emit(f"等待喚醒詞「{config.WAKE_WORD}」...")

        while self.running:
            try:
                self.signal_waveform.emit(True)
                wake_audio = utils.record_until_silence(self.assistant.models.vad, config.SILENCE_TIMEOUT_WAKE,
                                                        self.assistant.models.audio_device_index)
                self.signal_waveform.emit(False)

                wake_text = self.assistant.services.transcribe(wake_audio)
                if not utils.has_wake_word(wake_text): continue

                self.signal_status.emit("✅ 聽到了！請說話...")
                if config.WAKE_CONFIRM_FILE_ZH.exists(): utils.play_audio(str(config.WAKE_CONFIRM_FILE_ZH))

                self.signal_waveform.emit(True)
                self.camera_thread.start_capture_analysis()

                main_audio = utils.record_until_silence(self.assistant.models.vad, config.SILENCE_TIMEOUT_MAIN,
                                                        self.assistant.models.audio_device_index,
                                                        sync_face_capture=False)

                captured_frames = self.camera_thread.stop_capture_analysis()
                self.signal_waveform.emit(False)

                if main_audio is None:
                    self.signal_status.emit("沒聽到聲音...")
                    continue

                utils.play_filler_randomly()

                user_text = self.assistant.services.transcribe(main_audio)
                self.signal_user_text.emit(user_text)

                language = utils.detect_language(user_text)
                is_command, target = utils.check_personality_switch_command(user_text, language)
                if is_command:
                    switch_sound = config.PERSONALITY_SWITCH_FILE_ZH
                    if switch_sound.exists(): utils.play_audio(str(switch_sound))
                    self.assistant.manual_personality = target if target != 'reset' else None
                    reply = "好的！切換模式。"
                    self.signal_robot_text.emit(reply)

                    target_p = target if target else "humorous"
                    p_name = config.PERSONALITY_CONFIGS[target_p]['name']
                    self.signal_final_result.emit(f"🏆 綜合判定: {p_name} (手動)")

                    self.signal_speaking.emit(True, target_p)
                    self.assistant.services.text_to_speech_cosvoice(reply, config.REPLY_WAV, target_p)
                    self.signal_speaking.emit(False, target_p)
                    continue

                self.signal_status.emit("正在全模態分析情緒...")

                audio_tensor = torch.from_numpy(main_audio).float().unsqueeze(0) / 32768.0
                torchaudio.save(str(config.USER_AUDIO_WAV), audio_tensor, config.SAMPLE_RATE)

                future_text = self.executor.submit(self.assistant.services.detect_text_emotion, user_text, "zh")
                future_audio = self.executor.submit(self.assistant.services.detect_audio_emotion,
                                                    str(config.USER_AUDIO_WAV), "zh")
                future_face = self.executor.submit(self.assistant.services.analyze_facial_emotion_from_images,
                                                   captured_frames)
                future_body = self.executor.submit(self.assistant.services.detect_body_emotion, captured_frames)

                text_emo = future_text.result()
                audio_emo = future_audio.result()
                face_emo = future_face.result()
                body_emo = future_body.result()

                text_e = text_emo['emotion'] if text_emo else None
                audio_e = audio_emo['emotion'] if audio_emo else None
                face_e = face_emo['emotion'] if face_emo else None
                if text_e or audio_e or face_e:
                    self.assistant.memory.add_emotion(text_e, audio_e, face_e)

                gui_data = {
                    "text_score": text_emo['score'] if text_emo else 0.0,
                    "text_label": config.EMOTION_ZH_MAP.get(text_emo['emotion'],
                                                            text_emo['emotion']) if text_emo else "無",
                    "audio_score": audio_emo['score'] if audio_emo else 0.0,
                    "audio_label": config.EMOTION_ZH_MAP.get(audio_emo['emotion'],
                                                             audio_emo['emotion']) if audio_emo else "無",
                    "face_score": face_emo['confidence'] if face_emo else 0.0,
                    "face_label": face_emo['emotion_zh'] if face_emo else "無",
                    "body_score": body_emo['score'] if body_emo else 0.0,
                    "body_label": body_emo['label'] if body_emo else "無"
                }
                self.signal_emotions.emit(gui_data)

                self.signal_status.emit("思考中...")

                personality = self.assistant.manual_personality or self.assistant.services.select_personality_auto(
                    text_e, audio_e, face_e, self.assistant.memory,
                    text_score=text_emo['score'] if text_emo else 0.0,
                    audio_score=audio_emo['score'] if audio_emo else 0.0,
                    face_score=face_emo['confidence'] if face_emo else 0.0,
                    body_emo=body_emo['emotion'] if body_emo else None,
                    body_score=body_emo['score'] if body_emo else 0.0
                )

                p_name = config.PERSONALITY_CONFIGS[personality]['name']
                if self.assistant.manual_personality:
                    self.signal_final_result.emit(f"🏆 綜合判定: {p_name} (手動模式)")
                else:
                    final_emo = getattr(self.assistant.services, 'last_fused_emotion', 'neutral')
                    final_emo_zh = config.EMOTION_ZH_MAP.get(final_emo, final_emo)
                    self.signal_final_result.emit(f"🏆 當前情緒: {final_emo_zh}  ➔  決定人格: {p_name}")

                reply, updated_history = self.assistant.services.generate_response(
                    user_text, personality, self.assistant.conversation_history, text_emo, audio_emo, face_emo
                )
                self.assistant.conversation_history = updated_history
                self.signal_robot_text.emit(reply)

                if config.ENABLE_CONVERSATION_LOG:
                    turn_data = {
                        "turn_id": len(self.assistant.conversation_history) // 2,
                        "user_text": user_text, "bot_text": reply, "personality": personality,
                        "emotions": {"text": text_emo, "audio": audio_emo, "face": face_emo, "body": body_emo},
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    self.assistant.logger.log_turn(turn_data)

                self.signal_status.emit("說話中...")

                self.signal_speaking.emit(True, personality)
                self.assistant.services.text_to_speech_cosvoice(reply, config.REPLY_WAV, personality)
                self.signal_speaking.emit(False, personality)

                self.signal_status.emit(f"等待喚醒詞「{config.WAKE_WORD}」...")

            except Exception as e:
                print(f"❌ Worker Error: {e}")
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

        # 🚀 實例化頭像影片播放器
        self.avatar_thread = AvatarPlayerThread()
        self.avatar_thread.change_pixmap_signal.connect(self.update_avatar_image)

        self.worker = AIWorker(self.camera_thread)
        self.worker.signal_status.connect(self.update_status)
        self.worker.signal_user_text.connect(self.update_user_text)
        self.worker.signal_robot_text.connect(self.update_robot_text)
        self.worker.signal_emotions.connect(self.update_emotions)
        self.worker.signal_waveform.connect(self.update_waveform)

        self.worker.signal_speaking.connect(self.update_avatar_state)
        self.worker.signal_final_result.connect(self.update_final_result)

        self.worker.start()

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)

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

        self.lbl_cam = DynamicTextLabel("影像輸入", base_size=12,
                                        align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        left_layout.addWidget(self.lbl_cam, 1)

        self.camera_label = ScalableLabel()
        self.camera_label.setStyleSheet("background-color: black; border-radius: 10px;")
        left_layout.addWidget(self.camera_label, 8)

        self.user_text_box = DynamicTextLabel("等待輸入...", base_size=12, weight=QFont.Weight.Normal)
        self.user_text_box.setWordWrap(True)
        self.user_text_box.setStyleSheet(
            "background-color: #F0F0F0; border-radius: 10px; padding: 10px; color: #333;"
        )
        left_layout.addWidget(self.user_text_box, 2)

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

        lbl_emo = DynamicTextLabel("模態信心度", base_size=12,
                                   align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        emo_layout.addWidget(lbl_emo, 1)

        bars_layout = QVBoxLayout()
        self.bar_text = EmotionBar("文字", "#FFD700")
        self.bar_audio = EmotionBar("語音", "#87CEFA")
        self.bar_face = EmotionBar("人臉", "#90EE90")
        self.bar_body = EmotionBar("肢體", "#FF69B4")
        bars_layout.addWidget(self.bar_text)
        bars_layout.addWidget(self.bar_audio)
        bars_layout.addWidget(self.bar_face)
        bars_layout.addWidget(self.bar_body)
        bars_layout.addStretch()

        emo_layout.addLayout(bars_layout, 5)

        self.lbl_final_result = DynamicTextLabel("🏆 綜合判定: 等待分析...", base_size=12, weight=QFont.Weight.Bold)
        self.lbl_final_result.setStyleSheet(
            "color: #E65100; background-color: #FFF3E0; border-radius: 8px; padding: 5px;")
        emo_layout.addWidget(self.lbl_final_result, 1)

        # ==========================================
        # 機器人面板
        # ==========================================
        bot_panel = QFrame()
        bot_panel.setStyleSheet("background-color: transparent;")
        bot_layout = QVBoxLayout(bot_panel)

        # 🚀 保持使用單純的 ScalableLabel (配合穩定的 OpenCV 影片播放)
        self.bot_img = ScalableLabel()
        self._set_idle_avatar()

        # 🚀 [修改 1] 提高圖片/影片層的權重，讓它佔據 8 份空間 (原本是 6)
        bot_layout.addWidget(self.bot_img, 8)

        # 🚀 [修改 2] 將機器人回覆文字改回一般的 QLabel，固定字體大小避免失控放大
        self.bot_text = QLabel("...")
        self.bot_text.setFont(QFont("Microsoft JhengHei", 14, QFont.Weight.Bold))  # 固定大小為 14
        self.bot_text.setWordWrap(True)
        self.bot_text.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)  # 讓文字靠上對齊
        self.bot_text.setStyleSheet("color: #555; padding: 5px;")

        # 🚀 [修改 3] 降低文字框的權重，讓它只佔 2 份空間 (原本是 3)
        bot_layout.addWidget(self.bot_text, 2)

        right_layout.addWidget(emo_panel, 1)
        right_layout.addWidget(bot_panel, 1)

        content_layout.addWidget(left_panel, 1)
        content_layout.addWidget(right_panel, 1)
        main_layout.addLayout(content_layout)

    def _set_idle_avatar(self):
        """設定靜止狀態的機器人圖片"""
        idle_path = os.path.join("assets", "robot_avatar.png")

    def _set_idle_avatar(self):
        """設定靜止狀態的機器人圖片"""
        idle_path = os.path.join("assets", "robot_avatar.png")
        if os.path.exists(idle_path):
            self.bot_img.setPixmap(QPixmap(idle_path))
        else:
            self.bot_img.setText("Robot Idle")

    @pyqtSlot(QImage)
    def update_avatar_image(self, qt_image):
        """接收影片影格並更新 UI"""
        self.bot_img.setPixmap(QPixmap.fromImage(qt_image))

    @pyqtSlot(bool, str)
    def update_avatar_state(self, is_speaking, personality):
        """控制播放與停止"""
        if is_speaking:
            video_path = os.path.abspath(os.path.join("assets", f"{personality}.mp4"))
            if not os.path.exists(video_path):
                video_path = os.path.abspath(os.path.join("assets", "default.mp4"))
                if not os.path.exists(video_path):
                    return

                    # 啟動影片播放執行緒
            self.avatar_thread.play_video(video_path)
        else:
            # 停止播放並切回靜態圖片
            self.avatar_thread.stop_video()
            self._set_idle_avatar()

    @pyqtSlot(str)
    def update_final_result(self, text):
        self.lbl_final_result.setText(convert(text, 'zh-tw'))

    @pyqtSlot(QImage)
    def update_camera_image(self, qt_image):
        self.camera_label.setPixmap(QPixmap.fromImage(qt_image))

    def update_status(self, text):
        self.status_label.setText(convert(text, 'zh-tw'))

    def update_user_text(self, text):
        self.user_text_box.setText(convert(text, 'zh-tw'))

    def update_robot_text(self, text):
        self.bot_text.setText(convert(text, 'zh-tw'))

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
        self.avatar_thread.stop_video()
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