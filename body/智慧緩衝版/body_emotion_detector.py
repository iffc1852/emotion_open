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
            model_complexity=1,  # 1=普通 (適合 CPU), 0=Lite (最快), 2=Heavy (最準)
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.EMOTION_MAP = {
            "happy": "快樂 (肢體)",
            "surprise": "驚訝 (肢體)",
            "defensive": "防衛 (肢體)",
            "neutral": "中性 (肢體)"
        }

    def analyze_pose(self, image_paths):
        """
        分析多張圖片，回傳出現頻率最高的肢體情緒
        """
        if not image_paths:
            return {"emotion": "neutral", "score": 0.0, "label": "無"}

        detected_emotions = []

        for img_path in image_paths:
            frame = cv2.imread(img_path)
            if frame is None: continue

            # MediaPipe 需要 RGB
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(img_rgb)

            if results.pose_landmarks:
                emotion = self._heuristic_logic(results.pose_landmarks.landmark)
                detected_emotions.append(emotion)

        if not detected_emotions:
            return {"emotion": "neutral", "score": 0.0, "label": "無"}

        # 統計出現最多次的情緒
        most_common = max(set(detected_emotions), key=detected_emotions.count)
        # 計算出現比例作為信心度
        score = detected_emotions.count(most_common) / len(detected_emotions)

        # 簡單加權：如果有偵測到特殊動作，信心度給高一點 (讓它容易被選中)
        if most_common != "neutral":
            score = min(score + 0.3, 1.0)

        return {
            "emotion": most_common,
            "label": self.EMOTION_MAP.get(most_common, "未知"),
            "score": score
        }

    def _heuristic_logic(self, landmarks):
        """
        基於骨架座標的規則判斷邏輯
        """
        # 取得關鍵點 (Y座標：0是頂部，1是底部)
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # 1. 快樂/激動：雙手高舉 (手腕高於鼻子)
        if left_wrist.y < nose.y and right_wrist.y < nose.y:
            return "happy"

        # 2. 驚訝/恐懼：手在臉附近 (手腕高度在肩膀與頭頂之間)
        if (left_wrist.y < left_shoulder.y and left_wrist.y > nose.y) or \
                (right_wrist.y < right_shoulder.y and right_wrist.y > nose.y):
            return "surprise"

        # 3. 防衛/生氣：雙手抱胸 (判斷手腕X座標是否交叉且靠近肩膀高度)
        wrist_dist = abs(left_wrist.x - right_wrist.x)
        if wrist_dist < 0.2 and left_wrist.y > left_shoulder.y:
            return "defensive"

        return "neutral"