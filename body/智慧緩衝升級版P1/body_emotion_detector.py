# body_emotion_detector.py
import cv2
import mediapipe as mp
import numpy as np


class BodyLanguageDetector:
    def __init__(self):
        # 初始化 MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.EMOTION_MAP = {
            "happy": "快樂 (肢體)",
            "surprise": "驚訝 (肢體)",
            "defensive": "防衛 (肢體)",
            "neutral": "中性 (肢體)"
        }

    def analyze_pose(self, input_data):
        """
        🚀 [修正] 支援影像數據列表 (List[np.ndarray]) 或路徑列表 (List[str])
        """
        if not input_data:
            return {"emotion": "neutral", "score": 0.0, "label": "無"}

        detected_emotions = []

        for item in input_data:
            img_rgb = None

            # 判斷傳入的是路徑(str) 還是 影像數據(numpy array)
            if isinstance(item, str):
                # 如果是路徑，讀取檔案
                frame = cv2.imread(item)
                if frame is not None:
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elif isinstance(item, np.ndarray):
                # 🚀 如果是影像數據 (RGB)，直接用
                # 注意：CameraThread 傳出來的已經是 RGB 了，所以不需要再轉 BGR2RGB
                # 但為了保險，我們假設它是 RGB
                img_rgb = item

            if img_rgb is None: continue

            # MediaPipe 處理
            results = self.pose.process(img_rgb)

            if results.pose_landmarks:
                emotion = self._heuristic_logic(results.pose_landmarks.landmark)
                detected_emotions.append(emotion)

        if not detected_emotions:
            return {"emotion": "neutral", "score": 0.0, "label": "無"}

        most_common = max(set(detected_emotions), key=detected_emotions.count)
        score = detected_emotions.count(most_common) / len(detected_emotions)

        if most_common != "neutral":
            score = min(score + 0.3, 1.0)

        return {
            "emotion": most_common,
            "label": self.EMOTION_MAP.get(most_common, "未知"),
            "score": score
        }

    def _heuristic_logic(self, landmarks):
        """基於骨架座標的規則判斷邏輯"""
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # 1. 快樂/激動：雙手高舉
        if left_wrist.y < nose.y and right_wrist.y < nose.y:
            return "happy"

        # 2. 驚訝/恐懼：手在臉附近
        if (left_wrist.y < left_shoulder.y and left_wrist.y > nose.y) or \
                (right_wrist.y < right_shoulder.y and right_wrist.y > nose.y):
            return "surprise"

        # 3. 防衛/生氣：雙手抱胸
        wrist_dist = abs(left_wrist.x - right_wrist.x)
        if wrist_dist < 0.2 and left_wrist.y > left_shoulder.y:
            return "defensive"

        return "neutral"