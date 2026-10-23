from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "models" / "face_landmarker.task"


class FaceDetector:
    def __init__(self, model_path=DEFAULT_MODEL, num_faces=1):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Nu gasesc modelul la: {model_path}\n"
                "Pune face_landmarker.task in folderul models/."
            )

        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=num_faces,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts_ms = -1
        self.last_transform = None

    def detect(self, frame_bgr, timestamp_s):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        ts_ms = int(timestamp_s * 1000)
        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms

        result = self._landmarker.detect_for_video(mp_image, ts_ms)

        if not result.face_landmarks:
            self.last_transform = None
            return None

        if result.facial_transformation_matrixes:
            self.last_transform = np.array(result.facial_transformation_matrixes[0])

        return np.array(
            [[p.x, p.y, p.z] for p in result.face_landmarks[0]],
            dtype=np.float32,
        )

    def close(self):
        self._landmarker.close()