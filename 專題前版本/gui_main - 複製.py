# gui_main.py
import sys
import os
import cv2
import time
import gc
import threading
import queue
import re
import sounddevice as sd
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
# 🚀 [新增] 限制 CPU 執行緒，防止排程器打架拖慢 TTS
torch.set_num_threads(8)
import config
import utils
from main import MultimodalAssistant


os.environ['no_proxy'] = '127.0.0.1,localhost' #消滅 2 秒的網路握手延遲！
os.environ['NO_PROXY'] = '127.0.0.1,localhost' #消滅 2 秒的網路握手延遲！

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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
            target_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        x = (target_size.width() - scaled_pixmap.width()) // 2
        y = (target_size.height() - scaled_pixmap.height()) // 2
        painter.drawPixmap(x, y, scaled_pixmap)


#  OpenCV 影片播放執行緒
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
        if not self.cap.isOpened(): return

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps): fps = 24.0
        delay = 1.0 / fps

        while self.running:
            start_time = time.time()
            ret, frame = self.cap.read()

            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w

            qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
            self.change_pixmap_signal.emit(qt_img)

            elapsed = time.time() - start_time
            sleep_time = delay - elapsed
            if sleep_time > 0: time.sleep(sleep_time)

        self.cap.release()


#  輕量級視訊鏡頭 (MediaPipe 極速 68 點追蹤，零卡頓！)
class CameraThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self.running = True
        self.is_capturing_for_ai = False
        self.captured_frames = []
        self.last_save_time = 0

        # 🚀 [黃金對應矩陣] 將 MediaPipe 轉換為傳統的 68 個特徵點
        self.MP_68_INDICES = [
            162, 234, 93, 58, 172, 136, 149, 148, 152, 377, 378, 365, 397, 288, 323, 454, 389,  # 下巴輪廓
            70, 63, 105, 66, 107,  # 右眉毛
            336, 296, 334, 293, 300,  # 左眉毛
            168, 197, 5, 4,  # 鼻樑
            98, 97, 2, 326, 327,  # 鼻尖
            33, 160, 158, 133, 153, 144,  # 右眼
            362, 385, 387, 263, 373, 380,  # 左眼
            61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181,  # 外嘴唇
            78, 81, 13, 311, 308, 402, 14, 178  # 內嘴唇
        ]

        try:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.use_mesh = True
            print("\n   ✅ [高科技面部追蹤] MediaPipe 輕量載入成功！(零 CPU/GPU 負擔)")
        except Exception as e:
            print(f"   ⚠️ [高科技面部追蹤] 載入失敗 (請確認有執行 pip install mediapipe): {e}")
            self.use_mesh = False

    def run(self):
        # 1. 這裡改成老師給您的影片檔名 (確保影片跟 gui_main.py 放在同一個資料夾)
        cap = cv2.VideoCapture("01-01-04-01-01-01-01sad.mp4")
        if not cap.isOpened():
            print("⚠️ 找不到影片檔！請確認檔名正確且在同一個資料夾。")
            return

        while self.running:
            ret, cv_img = cap.read()

            # ==========================================
            # 🚀 關鍵修改：如果影片播完了 (ret 變成 False)，就讓它從頭開始！
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 把播放進度拉回第 0 幀
                continue  # 重新進入迴圈讀取畫面
            # ==========================================

            draw_img = cv_img.copy()

            #  瞬間畫點，完全不影響流暢度 (維持您原本的 MediaPipe 邏輯)
            if self.use_mesh:
                rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_img)

                if results.multi_face_landmarks:
                    h, w, _ = draw_img.shape
                    for face_landmarks in results.multi_face_landmarks:
                        # 按照矩陣，精準畫出那 68 個點
                        for idx in self.MP_68_INDICES:
                            pt = face_landmarks.landmark[idx]
                            x, y = int(pt.x * w), int(pt.y * h)
                            cv2.circle(draw_img, (x, y), 2, (0, 255, 0), -1)

            # 將 OpenCV 影像轉換成 PyQt 介面用的 QImage
            rgb_image = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
            self.change_pixmap_signal.emit(convert_to_Qt_format)

            if self.is_capturing_for_ai:
                current_time = time.time()
                #  嚴格 1 秒抓一張，上限設為 15 (足夠支撐 15 秒的對話)
                if (current_time - self.last_save_time >= 1.0) and (len(self.captured_frames) < 15):
                    self.last_save_time = current_time

                    small_img = cv2.resize(cv_img, (320, 240), interpolation=cv2.INTER_AREA)
                    rgb_small = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)
                    self.captured_frames.append(rgb_small)

            time.sleep(0.03)  # 控制播放速度約為 30 FPS

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

        self.lbl_name = DynamicTextLabel(label_text, base_size=28,
                                         align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.lbl_val = DynamicTextLabel("0%", base_size=28,
                                        align=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self.pbar = QProgressBar()
        self.pbar.setTextVisible(False)
        self.pbar.setMinimumHeight(8)
        self.pbar.setMaximumHeight(32)
        self.pbar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.pbar.setStyleSheet(
            f"QProgressBar {{ border: none; background-color: #E0E0E0; border-radius: 4px; }} QProgressBar::chunk {{ background-color: {color}; border-radius: 4px; }}")

        layout.addWidget(self.lbl_name, 6)
        layout.addWidget(self.pbar, 10)
        layout.addWidget(self.lbl_val, 6)

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
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)

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

                # 🚀 關鍵修改：加上 force_language="zh"，強制用中文辨識喚醒詞
                wake_text = self.assistant.services.transcribe(wake_audio, force_language="zh")

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
                self.signal_status.emit("正在同步聽打與感知情緒 (CPU/GPU 協同)...")

                # 先把聲音存檔，給 Emotion2Vec+ 讀取
                audio_tensor = torch.from_numpy(main_audio).float().unsqueeze(0) / 32768.0
                torchaudio.save(str(config.USER_AUDIO_WAV), audio_tensor, config.SAMPLE_RATE)

                # ==========================================
                # 🚀 [多核平行運算啟動] 讓 CPU 和 GPU 同時開工！
                # ==========================================
                # 1. 派發聽打任務 (交給 i7 CPU 處理 Whisper)
                future_text_transcribe = self.executor.submit(self.assistant.services.transcribe, main_audio)

                # 2. 派發語音情緒任務 (交給 GPU 或 CPU 處理 Emotion2Vec)
                future_audio = self.executor.submit(self.assistant.services.detect_audio_emotion,
                                                    str(config.USER_AUDIO_WAV), "zh")

                # 3. 派發人臉情緒任務 (交給 i7 CPU 處理 Py-FEAT)
                future_face = self.executor.submit(self.assistant.services.analyze_facial_emotion_from_images,
                                                   captured_frames)

                # ==========================================
                # 等待「聽打結果」出來 (因為 LLM 回答必須要有文字)
                # ==========================================
                user_text = future_text_transcribe.result()
                self.signal_user_text.emit(user_text)

                if not user_text or not user_text.strip():
                    self.signal_status.emit(f"等待喚醒詞「{config.WAKE_WORD}」...")
                    continue

                language = utils.detect_language(user_text)
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
                    self.signal_final_result.emit(f"🏆 綜合判定: {p_name} (手動模式)")

                    self.signal_speaking.emit(True, target_p)
                    self.assistant.services.text_to_speech_cosvoice(reply, config.REPLY_WAV, target_p)
                    self.signal_speaking.emit(False, target_p)
                    continue

                # ==========================================
                # 4. 文字出來後，立刻派發「文字情緒」任務
                # ==========================================
                self.signal_status.emit("正在分析文字情緒...")
                future_text_emo = self.executor.submit(self.assistant.services.detect_text_emotion, user_text, language)

                # ==========================================
                # 5. 一次收割所有成果
                # (因為平行運算，此時語音和人臉情緒早就已經算完在等您了)
                # ==========================================
                text_emo = future_text_emo.result()
                audio_emo = future_audio.result()
                face_emo = future_face.result()

                text_e = text_emo['emotion'] if text_emo else None
                audio_e = audio_emo['emotion'] if audio_emo else None
                face_e = face_emo['emotion'] if face_emo else None

                # 只有在有新情緒時，才寫入記憶體
                if text_e or audio_e or face_e:
                    self.assistant.memory.add_emotion(text_e, audio_e, face_e)

                # ==========================================
                # 🚀 [UI 美化與修復] 將函式獨立，確保資料傳遞安全
                # ==========================================
                def clean_for_ui(emo_dict):
                    if not emo_dict:
                        return "無"

                    raw = str(emo_dict.get("emotion", ""))

                    if "/" in raw:
                        raw = raw.split("/")[0].strip()

                    return config.EMOTION_ZH_MAP.get(raw, raw)

                def get_normalized_emotion_key(emo_dict):
                    if not emo_dict:
                        return None

                    raw = str(emo_dict.get("emotion", "")).lower()

                    if "/" in raw:
                        raw = raw.split("/")[0].strip()

                    return config.EMOTION_NORMALIZATION.get(raw, raw)

                def get_weighted_ui_score(emo_dict, weight_table):
                    if not emo_dict:
                        return 0.0

                    emotion_key = get_normalized_emotion_key(emo_dict)
                    raw_score = float(emo_dict.get("score", 0.0))
                    weight = weight_table.get(emotion_key, 1.0)

                    return min(1.0, max(0.0, raw_score * weight))

                # ===== 只加權一次：先算出三種模態的最終分數 =====

                text_weighted_score = get_weighted_ui_score(
                    text_emo,
                    getattr(config, "EMOTION_TEXT_WEIGHTS", {})
                )

                audio_weighted_score = get_weighted_ui_score(
                    audio_emo,
                    getattr(config, "EMOTION_AUDIO_WEIGHTS", {})
                )

                face_weighted_score = (
                    face_emo.get("adjusted_confidence", face_emo.get("confidence", 0.0))
                    if face_emo else 0.0
                )

                # ===== UI 顯示使用加權後分數 =====

                gui_data = {
                    "text_score": text_weighted_score,
                    "text_label": clean_for_ui(text_emo),

                    "audio_score": audio_weighted_score,
                    "audio_label": clean_for_ui(audio_emo),

                    "face_score": face_weighted_score,
                    "face_label": face_emo["emotion_zh"] if face_emo else "無",
                }

                self.signal_emotions.emit(gui_data)
                # ==========================================

                self.signal_status.emit("思考中...")
                personality = self.assistant.manual_personality or self.assistant.services.select_personality_auto(
                    text_e, audio_e, face_e, self.assistant.memory,
                    text_score=text_weighted_score,
                    audio_score=audio_weighted_score,
                    face_score=face_weighted_score,
                    body_emo=None,
                    body_score=0.0
                )

                p_name = config.PERSONALITY_CONFIGS[personality]['name']
                if self.assistant.manual_personality:
                    self.signal_final_result.emit(f"🏆 綜合判定: {p_name} (手動模式)")
                else:
                    final_emo = getattr(self.assistant.services, 'last_fused_emotion', 'neutral')
                    final_emo_zh = config.EMOTION_ZH_MAP.get(final_emo, final_emo)
                    self.signal_final_result.emit(f"🏆 當前情緒: {final_emo_zh}  ➔  決定人格: {p_name}")

                self.signal_status.emit("回應中...")
                self.signal_speaking.emit(True, personality)
                self.signal_robot_text.emit("")

                t_start_pipeline = time.time()

                print(f"\n   {'=' * 55}")
                print(f"   🚀 [序列化接力] 啟動全速模式")
                print(f"   {'=' * 55}")

                # ==========================================
                # 第一階段：LLM 全力衝刺 (非串流，一次拿滿)
                # ==========================================
                print(f"   [🧠 LLM 生成] 正在全速生成回應文字...")

                # 這裡改用 services.py 裡原本的 generate_response (如果沒改過，預設就是非串流的)
                # 如果 services 裡已經全改成 stream=True，請確保有這個非串流的函式
                full_reply, _ = self.assistant.services.generate_response(
                    user_text, personality, self.assistant.conversation_history, text_e, audio_e, face_e, language
                )

                t_llm_finish = time.time()
                print(f"   [🧠 LLM 完工] 耗時: {t_llm_finish - t_start_pipeline:.2f}s ➔ 「{full_reply}」")

                # LLM 算完後，一次性把整段文字顯示在畫面上
                self.signal_robot_text.emit(full_reply)

                # ==========================================
                # 第二階段：CosyVoice 全力衝刺 (保證長度，音色最佳)
                # ==========================================
                if full_reply:
                    print(f"   [⚙️ TTS 合成] 正在進行高保真語音合成...")
                    t_tts_start = time.time()


                    # 將整段話交給 CosyVoice，避免 too short 問題
                    audio_data, cost_time = self.assistant.services.synthesize_cosvoice_audio(full_reply, personality)

                    if audio_data is not None:
                        print(f"   [⚙️ TTS 完工] 耗時: {cost_time:.2f}s ➔ 準備播放")

                        # ==========================================
                        # 第三階段：播放語音並處理行政雜務
                        # ==========================================
                        print(f"   [🔊 語音播放] ▶️ 開始播放")

                        # 利用 threading 讓播放不卡死介面，同時更新歷史紀錄
                        def play_and_update():
                            sd.play(audio_data, samplerate=22050, blocking=True)

                        play_thread = threading.Thread(target=play_and_update)
                        play_thread.start()

                        # 趁著語音正在播，把這輪對話寫入記憶
                        print(f"   [💾 紀錄搬運] 語音播放中，同步更新對話歷史與日誌...")
                        self.assistant.conversation_history.append({"role": "user", "content": user_text})
                        self.assistant.conversation_history.append({"role": "assistant", "content": full_reply})
                        # 只保留最近 10 則訊息，避免長時間使用 RAM 慢慢膨脹
                        self.assistant.conversation_history = self.assistant.conversation_history[-10:]
                        # 🚀 [修復] 復活日誌系統！把這輪的完整多模態數據寫入 Logs 資料夾
                        turn_data = {
                            "user_text": user_text,
                            "reply": full_reply,
                            "text_emotion": text_e,
                            "audio_emotion": audio_e,
                            "facial_emotion": face_e,
                            "personality": personality,
                            "personality_mode": "manual" if self.assistant.manual_personality else "auto"
                        }
                        self.assistant.logger.log_turn(turn_data)
                        # 👆 寫入完成！

                        play_thread.join()
                        print(f"   [🔊 語音播放] ⏹️ 播放結束")

                        # 播放結束後再清理本輪大型暫存資料
                        try:
                            del audio_data
                        except NameError:
                            pass

                        gc.collect()

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    else:
                        print(f"   [⚠️ TTS 失敗] 無法產生語音")
                        self.assistant.conversation_history.append({"role": "user", "content": user_text})
                        self.assistant.conversation_history.append({"role": "assistant", "content": full_reply})
                        # 只保留最近 10 則訊息，避免長時間使用 RAM 慢慢膨脹
                        self.assistant.conversation_history = self.assistant.conversation_history[-10:]
                else:
                    print("   [⚠️ LLM 空白] 未產生任何文字")

                print(f"   {'=' * 55}")
                print(f"   🏁 [接力完成] 總耗時: {time.time() - t_start_pipeline:.2f}s")
                print(f"   {'=' * 55}\n")

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
        self.resize(1100, 700)
        self.setStyleSheet("background-color: #F5F5F7;")

        self.camera_thread = CameraThread()
        self.camera_thread.change_pixmap_signal.connect(self.update_camera_image)
        self.camera_thread.start()

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

        left_panel = QFrame()
        left_panel.setStyleSheet("QFrame { background-color: white; border-radius: 15px; border: 1px solid #E0E0E0; }")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(15, 15, 15, 15)

        self.lbl_cam = DynamicTextLabel("影像輸入", base_size=14,
                                        align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        left_layout.addWidget(self.lbl_cam, 1)

        self.camera_label = ScalableLabel()
        self.camera_label.setStyleSheet("background-color: black; border-radius: 10px;")
        left_layout.addWidget(self.camera_label, 3)

        self.wave_label = QLabel("〰️〰️〰️")
        self.wave_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.wave_label.setStyleSheet("color: #AAA; font-size: 20px;")
        self.wave_label.setFixedHeight(30)
        left_layout.addWidget(self.wave_label, 1)

        emo_panel = QFrame()
        emo_panel.setStyleSheet("QFrame { background-color: transparent; border: none; }")
        emo_layout = QVBoxLayout(emo_panel)
        emo_layout.setContentsMargins(0, 0, 0, 0)

        lbl_emo = DynamicTextLabel("模態信心度", base_size=38,
                                   align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        emo_layout.addWidget(lbl_emo)

        self.bar_text = EmotionBar("文字", "#FFD700")
        self.bar_audio = EmotionBar("語音", "#87CEFA")
        self.bar_face = EmotionBar("人臉", "#90EE90")

        emo_layout.addWidget(self.bar_text)
        emo_layout.addWidget(self.bar_audio)
        emo_layout.addWidget(self.bar_face)
        emo_layout.addStretch()

        self.lbl_final_result = DynamicTextLabel("🏆 綜合判定: 等待分析...", base_size=32, weight=QFont.Weight.Bold)
        self.lbl_final_result.setStyleSheet(
            "color: #E65100; background-color: #FFF3E0; border-radius: 8px; padding: 5px;")
        emo_layout.addWidget(self.lbl_final_result)

        left_layout.addWidget(emo_panel, 7)

        right_panel = QFrame()
        right_panel.setStyleSheet("background-color: transparent;")
        right_layout = QVBoxLayout(right_panel)

        self.user_text_box = QLabel("等待輸入...")
        self.user_text_box.setFont(QFont("Microsoft JhengHei", 32, QFont.Weight.Normal))
        self.user_text_box.setWordWrap(True)
        self.user_text_box.setStyleSheet(
            "background-color: #E3F2FD; border-radius: 10px; padding: 15px; color: #333; border: 1px solid #BBDEFB;")
        self.user_text_box.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        right_layout.addWidget(self.user_text_box, 2)

        self.bot_img = ScalableLabel()
        self._set_idle_avatar()
        right_layout.addWidget(self.bot_img, 6)

        self.bot_text = QLabel("...")
        self.bot_text.setFont(QFont("Microsoft JhengHei", 32, QFont.Weight.Bold))
        self.bot_text.setWordWrap(True)
        self.bot_text.setStyleSheet(
            "background-color: #FFF3E0; border-radius: 10px; padding: 15px; color: #555; border: 1px solid #FFE0B2;")
        self.bot_text.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        right_layout.addWidget(self.bot_text, 2)

        content_layout.addWidget(left_panel, 3)
        content_layout.addWidget(right_panel, 5)
        main_layout.addLayout(content_layout)

    def _set_idle_avatar(self):
        idle_path = os.path.join("assets", "robot_avatar.png")
        if os.path.exists(idle_path):
            self.bot_img.setPixmap(QPixmap(idle_path))
        else:
            self.bot_img.setText("Robot Idle")

    @pyqtSlot(QImage)
    def update_avatar_image(self, qt_image):
        self.bot_img.setPixmap(QPixmap.fromImage(qt_image))

    @pyqtSlot(bool, str)
    def update_avatar_state(self, is_speaking, personality):
        if is_speaking:
            video_path = os.path.abspath(os.path.join("assets", f"{personality}.mp4"))
            if not os.path.exists(video_path):
                video_path = os.path.abspath(os.path.join("assets", "default.mp4"))
                if not os.path.exists(video_path): return
            self.avatar_thread.play_video(video_path)
        else:
            self.avatar_thread.stop_video()
            self._set_idle_avatar()

    @pyqtSlot(str)
    def update_final_result(self, text):
        self.lbl_final_result.setText(convert(text, 'zh-tw'))

    @pyqtSlot(QImage)
    def update_camera_image(self, qt_image):
        self.camera_label.setPixmap(QPixmap.fromImage(qt_image))

    @pyqtSlot(str)
    def update_status(self, text):
        self.status_label.setText(convert(text, 'zh-tw'))

    @pyqtSlot(str)
    def update_user_text(self, text):
        # 🚀 自動偵測，中文才轉繁體
        if utils.detect_language(text) == "zh":
            text = convert(text, 'zh-tw')
        self.user_text_box.setText(f"👤 您說：{text.strip()}")

    @pyqtSlot(str)
    def update_robot_text(self, text):
        # 🚀 自動偵測，中文才轉繁體
        if utils.detect_language(text) == "zh":
            text = convert(text, 'zh-tw')
        self.bot_text.setText(f"🤖 小白：{text.strip()}")

    @pyqtSlot(dict)
    def update_emotions(self, data):
        self.bar_text.set_value(data.get('text_score', 0), data.get('text_label'))
        self.bar_audio.set_value(data.get('audio_score', 0), data.get('audio_label'))
        self.bar_face.set_value(data.get('face_score', 0), data.get('face_label'))

    @pyqtSlot(bool)
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