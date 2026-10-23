import time
import cv2

class Camera:
    def __init__(self, source=0,width=640, height=480,flip=None):
        self.is_file = isinstance(source, str)
        #source =0 - webcam
        #source =1 - camera externa
        #"cale.mp4" - fisier video

        self.flip = (not self.is_file) if flip is None else flip
        #oglidesc imaginea doar daca e live  

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

        if not self.is_file:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            self._t0 = time.monotonic()

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None, None

        if self.flip: 
            frame = cv2.flip(frame, 1)

        if self.is_file:
            timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        else:
            timestamp = time.monotonic() - self._t0
        return frame, timestamp

    def release(self):
        self.cap.release()

    def __enter__(self):
        return self 

    def __exit__(self, *args):
        self.release()