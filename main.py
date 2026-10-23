import time
import cv2

from src.capture.camera import Camera
from src.landmarks.face_detector import FaceDetector


def draw_landmarks(frame, landmarks):
    h, w = frame.shape[:2]
    points = (landmarks[:, :2] * [w, h]).astype(int)
    for x, y in points:
        cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)


def main():
    detector = FaceDetector()
    prev = time.perf_counter()

    with Camera(0) as cam:
        while True:
            frame, ts = cam.read()
            if frame is None:
                break

            landmarks = detector.detect(frame, ts)
            if landmarks is not None:
                draw_landmarks(frame, landmarks)
            else:
                cv2.putText(frame, "Fata nedetectata", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            now = time.perf_counter()
            fps = 1.0 / (now - prev)
            prev = now
            cv2.putText(frame, f"FPS: {fps:.0f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("Face Mesh", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()