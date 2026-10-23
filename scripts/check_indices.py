import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from src.capture.camera import Camera
from src.features.ear import LEFT_EYE, RIGHT_EYE
from src.features.geometry import to_pixels
from src.features.head_pose import POSE_INDICES
from src.features.mar import LOWER_LIP, MOUTH_LEFT, MOUTH_RIGHT, UPPER_LIP
from src.landmarks.face_detector import FaceDetector

GROUPS = [
    (RIGHT_EYE, (0, 255, 0)),
    (LEFT_EYE, (255, 200, 0)),
    ([UPPER_LIP, LOWER_LIP, MOUTH_LEFT, MOUTH_RIGHT], (0, 0, 255)),
    (POSE_INDICES, (0, 255, 255)),
]


def main():
    detector = FaceDetector()
    with Camera(0) as cam:
        while True:
            frame, ts = cam.read()
            if frame is None:
                break
            landmarks = detector.detect(frame, ts)
            if landmarks is not None:
                h, w = frame.shape[:2]
                points = to_pixels(landmarks, w, h)
                for indices, color in GROUPS:
                    for idx in indices:
                        x, y = int(points[idx][0]), int(points[idx][1])
                        cv2.circle(frame, (x, y), 2, color, -1)
                        cv2.putText(frame, str(idx), (x + 3, y - 3),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            cv2.imshow("Verificare indici", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()