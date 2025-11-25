"""
人臉情緒辨識模組 (Py-FEAT 版本) - 修正 Tensor Gradient 錯誤版

- 加入 torch.no_grad() 解決 RuntimeError
- 修正 nan% 錯誤
"""

import cv2
import numpy as np
import torch  # <--- 新增：必須匯入 torch
from pathlib import Path
from typing import Optional, Dict, List
import warnings
import os
import time
from datetime import datetime

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

    def __init__(self, face_model: str = "retinaface",
                 emotion_model: str = "resmasknet",
                 device: str = "cpu"):
        if not PYFEAT_AVAILABLE:
            raise ImportError("Py-FEAT 未安裝")

        print(f"🎭 初始化 Py-FEAT 分析模組...")
        print(f"   人臉模型: {face_model}")
        print(f"   情緒模型: {emotion_model}")
        print(f"   運算裝置: {device}")

        self.detector = Detector(
            face_model=face_model,
            landmark_model="mobilenet",
            au_model="xgb",
            emotion_model=emotion_model,
            facepose_model="img2pose",
            device=device
        )
        print(f"✅ 初始化完成")

    def detect_emotion_from_image_path(self, image_path: str,
                                       min_confidence: float = 0.3) -> Optional[Dict]:
        """從圖片路徑偵測情緒 (單一圖片)"""
        try:
            # 🟢 關鍵修正：使用 torch.no_grad() 禁用梯度計算
            # 這能解決 "Can't call numpy() on Tensor that requires grad" 錯誤
            with torch.no_grad():
                result_df = self.detector.detect_image(image_path)

            if result_df is None or len(result_df) == 0:
                print(f"    [Py-FEAT] ⚠️  未偵測到人臉: {os.path.basename(image_path)}")
                return None

            # 取得第一張臉的數據
            emotions = result_df.iloc[0]

            all_probs = {}
            for emotion_name in self.PYFEAT_EMOTION_COLUMNS:
                if emotion_name in emotions:
                    value = float(emotions[emotion_name])
                    if np.isnan(value):
                        # print(f"    [Py-FEAT] ⚠️  模型回傳 'nan' (非數值) for {emotion_name}")
                        value = 0.0
                    all_probs[emotion_name] = value

            if not all_probs:
                print(f"    [Py-FEAT] ⚠️  無法取得情緒分數")
                return None

            max_emotion = max(all_probs, key=all_probs.get)
            confidence = all_probs[max_emotion]

            debug_msg = f"    [Py-FEAT] 🔍 預測: {max_emotion} ({confidence:.2%})"
            if confidence < min_confidence:
                debug_msg += f" (信心度過低 < {min_confidence:.0%})"
                print(debug_msg)
                return None

            print(debug_msg)
            emotion_zh = self.EMOTION_ZH_MAP.get(max_emotion, max_emotion)

            return {
                "emotion": max_emotion,
                "emotion_zh": emotion_zh,
                "confidence": confidence,
                "all_probabilities": all_probs
            }

        except Exception as e:
            print(f"    [Py-FEAT] ⚠️  處理失敗: {e}")
            # 只有在 debug 模式下才印出完整錯誤，避免刷屏
            # import traceback
            # traceback.print_exc()
            return None

    def analyze_images(self,
                       image_paths: List[str],
                       min_confidence: float = 0.3,
                       frame_weights: Dict[str, float] = None) -> Optional[Dict]:
        """
        分析一組圖片路徑，並回傳加權平均情緒
        """
        if not image_paths:
            return None

        print(f"    [Py-FEAT] 🎞️  收到 {len(image_paths)} 幀影像進行分析...")

        frame_results = []
        for idx, img_path in enumerate(image_paths):
            result = self.detect_emotion_from_image_path(img_path, min_confidence)
            if result:
                result['frame_index'] = idx
                frame_results.append(result)

        if not frame_results:
            print(f"    [Py-FEAT] ⚠️  所有幀都未檢測到有效情緒")
            return None

        final_result = self._calculate_weighted_emotion(
            frame_results,
            frame_weights,
            total_frames=len(image_paths)
        )
        return final_result

    def _calculate_weighted_emotion(self,
                                    frame_results: List[Dict],
                                    custom_weights: Dict[str, float] = None,
                                    total_frames: int = 1) -> Optional[Dict]:
        """
        計算加權平均的最終情緒
        """
        if not frame_results:
            return None

        if custom_weights is None:
            custom_weights = {}

        emotion_scores = {}

        num_valid_frames = len(frame_results)

        for frame_idx, frame_result in enumerate(frame_results):
            all_probs = frame_result.get('all_probabilities', {})

            for emotion, prob in all_probs.items():
                weight = custom_weights.get(emotion, 1.0)
                weighted_score = prob * weight
                if emotion not in emotion_scores:
                    emotion_scores[emotion] = 0
                emotion_scores[emotion] += weighted_score

        if not emotion_scores:
            return None

        sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)

        # 計算平均信心度
        final_emotion = sorted_emotions[0][0]
        # 簡單平均：總分 / 有效幀數
        avg_confidence = sorted_emotions[0][1] / num_valid_frames

        all_probabilities = {
            emotion: score / num_valid_frames
            for emotion, score in emotion_scores.items()
        }

        print(f"    [Py-FEAT] 🎯 最終情緒: {self.EMOTION_ZH_MAP.get(final_emotion, final_emotion)} ({avg_confidence:.2%})")

        return {
            "emotion": final_emotion,
            "emotion_zh": self.EMOTION_ZH_MAP.get(final_emotion, final_emotion),
            "confidence": avg_confidence,
            "all_probabilities": all_probabilities,
            "num_valid_frames": num_valid_frames,
            "num_total_frames": total_frames,
            "frame_details": frame_results
        }


# ===== 全域函數 =====
_facial_detector_instance = None

def initialize_facial_emotion_detector(face_model: str = "retinaface",
                                       emotion_model: str = "resmasknet",
                                       device: str = "cpu") -> bool:
    """
    初始化人臉情緒辨識器(只需呼叫一次)
    """
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
        _facial_detector_instance = None
        return False

def analyze_images(image_paths: List[str],
                   min_confidence: float = 0.3,
                   frame_weights: Dict[str, float] = None) -> Optional[Dict]:
    """
    (公開函數) 分析一組已拍攝的圖片
    """
    global _facial_detector_instance

    if _facial_detector_instance is None:
        # 嘗試自動初始化 (如果尚未初始化)
        print("⚠️  Py-FEAT 尚未初始化，嘗試自動初始化...")
        if not initialize_facial_emotion_detector():
            return None

    try:
        result = _facial_detector_instance.analyze_images(
            image_paths, min_confidence, frame_weights
        )
        return result
    except Exception as e:
        print(f"⚠️  多幀分析失敗: {e}")
        return None
