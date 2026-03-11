# facial_emotion_detector_pyfeat.py

"""
人臉情緒辨識模組 (Py-FEAT 版本) - 極速記憶體版
整合了直接記憶體推論，並保留暫存檔作為備案
"""

import cv2
import numpy as np
import warnings
import os
import torch
import tempfile
from typing import Optional, Dict, List
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    from feat import Detector

    PYFEAT_AVAILABLE = True
except ImportError:
    PYFEAT_AVAILABLE = False
    print("⚠️  Py-FEAT 未安裝")


class FacialEmotionDetectorPyFeat:
    EMOTION_ZH_MAP = {
        "anger": "憤怒", "angry": "憤怒",
        "disgust": "厭惡",
        "fear": "恐懼",
        "happiness": "快樂", "happy": "快樂",
        "sadness": "悲傷", "sad": "悲傷",
        "surprise": "驚訝",
        "neutral": "中性"
    }

    PYFEAT_EMOTION_COLUMNS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']

    def __init__(self, face_model: str = "faceboxes",
                 emotion_model: str = "resmasknet",
                 device: str = "cpu"):
        if not PYFEAT_AVAILABLE:
            raise ImportError("Py-FEAT 未安裝")

        print(f"🎭 初始化 Py-FEAT 分析模組...")
        print(f"   運算裝置: {device.upper()}")

        self.detector = Detector(
            face_model=face_model,
            landmark_model="mobilenet",
            au_model="xgb",
            emotion_model=emotion_model,
            facepose_model="img2pose",
            device=device
        )
        print(f"✅ 初始化完成")

    def detect_emotion_from_memory_fast(self, frame_rgb: np.ndarray) -> Optional[Dict]:
        """
        繞過硬碟 I/O，直接呼叫模型底層 API
        """
        try:
            with torch.no_grad():
                # Py-FEAT 的底層 API 通常預期輸入是一個 List [frame]
                # 即使只有一張圖，也要包成 list，否則維度會錯
                batch_frames = [frame_rgb]

                # 1. 偵測人臉 (回傳的是 list of list of boxes)
                detected_faces = self.detector.detect_faces(batch_frames)

                # 檢查是否有偵測到人臉 (格式通常是 [[box1, box2...]])
                if not detected_faces or len(detected_faces) == 0 or len(detected_faces[0]) == 0:
                    return None

                # 2. 偵測特徵點
                landmarks = self.detector.detect_landmarks(batch_frames, detected_faces)

                # 3. 偵測情緒
                # output 通常是 list of arrays
                emotions = self.detector.detect_emotions(batch_frames, detected_faces, landmarks)

                # 解析結果 (取第一張圖、第一個臉)
                # emotions 結構通常是 [Batch][Face][Emotions]
                if emotions and len(emotions) > 0 and len(emotions[0]) > 0:
                    emo_scores = emotions[0][0]  # 第一張圖的第一個人臉

                    # 轉成字典
                    all_probs = {}
                    for idx, col in enumerate(self.PYFEAT_EMOTION_COLUMNS):
                        # Py-FEAT 版本不同，有時候回傳 tensor 有時候是 numpy
                        val = emo_scores[idx]
                        if hasattr(val, 'item'): val = val.item()
                        all_probs[col] = float(val)

                    max_emotion = max(all_probs, key=all_probs.get)
                    confidence = all_probs[max_emotion]
                    emotion_zh = self.EMOTION_ZH_MAP.get(max_emotion, max_emotion)

                    return {
                        "emotion": max_emotion,
                        "emotion_zh": emotion_zh,
                        "confidence": confidence,
                        "all_probabilities": all_probs
                    }
            return None
        except Exception as e:
            # 這裡不印錯誤，因為失敗了會自動跳轉去用暫存檔法
            # print(f"    [Fast Mode Failed] {e}")
            return None

    def detect_emotion_fallback(self, frame_rgb: np.ndarray) -> Optional[Dict]:
        """
        🐢 [備用版] 暫存檔大法 (當極速版失敗時使用)
        """
        temp_path = None
        try:
            fd, temp_path = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            # 轉回 BGR 存檔
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(temp_path, frame_bgr)

            with torch.no_grad():
                # 使用最外層的高階 API
                result_df = self.detector.detect_image(temp_path)

            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

            if result_df is None or len(result_df) == 0: return None

            emotions = result_df.iloc[0]
            all_probs = {}
            for emotion_name in self.PYFEAT_EMOTION_COLUMNS:
                if emotion_name in emotions:
                    value = float(emotions[emotion_name])
                    if np.isnan(value): value = 0.0
                    all_probs[emotion_name] = value

            if not all_probs: return None

            max_emotion = max(all_probs, key=all_probs.get)
            confidence = all_probs[max_emotion]
            emotion_zh = self.EMOTION_ZH_MAP.get(max_emotion, max_emotion)

            return {
                "emotion": max_emotion,
                "emotion_zh": emotion_zh,
                "confidence": confidence,
                "all_probabilities": all_probs
            }
        except Exception as e:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            return None

    def analyze_frames(self, frames: List[np.ndarray], min_confidence: float = 0.3) -> Optional[Dict]:
        """主入口：優先嘗試極速版，失敗則用備用版"""
        if not frames: return None

        frame_results = []
        for idx, frame in enumerate(frames):
            # 1. 優先嘗試極速記憶體模式
            result = self.detect_emotion_from_memory_fast(frame)

            # 2. 如果失敗 (回傳 None)，嘗試備用模式 (暫存檔)
            if not result:
                # print("    ⚠️ 切換至暫存檔模式")
                result = self.detect_emotion_fallback(frame)

            if result:
                result['frame_index'] = idx
                frame_results.append(result)

        if not frame_results: return None

        return self._calculate_weighted_emotion(frame_results, total_frames=len(frames))

    def _calculate_weighted_emotion(self, frame_results: List[Dict], total_frames: int = 1) -> Optional[Dict]:
        if not frame_results: return None

        emotion_scores = {}
        num_valid_frames = len(frame_results)

        for res in frame_results:
            all_probs = res.get('all_probabilities', {})
            for emotion, prob in all_probs.items():
                if emotion not in emotion_scores: emotion_scores[emotion] = 0
                emotion_scores[emotion] += prob

        if not emotion_scores: return None

        # 🚀 [新增] 讀取 config 裡面的人臉情緒專屬權重 (例如壓制容易誤判的 surprise=0.7)
        import config
        face_weights = getattr(config, 'FACIAL_EMOTION_WEIGHTS', {})
        adjusted_scores = {}
        for emo, raw_score in emotion_scores.items():
            w = face_weights.get(emo, 1.0)
            adjusted_scores[emo] = raw_score * w

        # 🚀 改用「調整後的分數」來選出最可能的情緒
        sorted_emotions = sorted(adjusted_scores.items(), key=lambda x: x[1], reverse=True)
        final_emotion = sorted_emotions[0][0]

        # 信心度依然回報原始的平均值，這樣 UI 顯示才準確
        final_score = emotion_scores[final_emotion]
        avg_confidence = final_score / num_valid_frames

        all_probabilities = {k: v / num_valid_frames for k, v in emotion_scores.items()}

        print(
            f"    [Py-FEAT] 🎯 最終情緒: {self.EMOTION_ZH_MAP.get(final_emotion, final_emotion)} ({avg_confidence:.2%})")

        return {
            "emotion": final_emotion,
            "emotion_zh": self.EMOTION_ZH_MAP.get(final_emotion, final_emotion),
            "confidence": avg_confidence,
            "all_probabilities": all_probabilities,
            "num_valid_frames": num_valid_frames,
            "num_total_frames": total_frames
        }

# ===== 全域函數 =====
_facial_detector_instance = None


def initialize_facial_emotion_detector(face_model: str = "faceboxes",
                                       emotion_model: str = "resmasknet",
                                       device: str = "cpu") -> bool:
    global _facial_detector_instance
    try:
        _facial_detector_instance = FacialEmotionDetectorPyFeat(
            face_model=face_model,
            emotion_model=emotion_model,
            device=device
        )
        return True
    except Exception as e:
        print(f"❌ 初始化失敗: {e}")
        return False


def analyze_frames_from_memory(frames: List[np.ndarray], min_confidence: float = 0.3) -> Optional[Dict]:
    global _facial_detector_instance
    if _facial_detector_instance is None:
        return None
    return _facial_detector_instance.analyze_frames(frames, min_confidence)


if __name__ == "__main__":
    print("Py-FEAT 模組載入成功")