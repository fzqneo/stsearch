import cv2
import numpy as np

from stsearch.interval import ImageInterval, RGB_KEY
import threading

class VideoFrameInterval(ImageInterval):

    def __init__(self, bounds, payload=dict(), root_decoder=None):
        super().__init__(bounds, payload=payload, root=None)
        self.root_decoder = root_decoder

    @property
    def rgb(self):
        if RGB_KEY not in self.payload:
            self.rgb = self.root_decoder.get_frame(self['t1'])
        return self.rgb

    @rgb.setter 
    def rgb(self, val):
        self.payload[RGB_KEY] = np.array(val)


class LocalVideoDecoder(object):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.cap = cv2.VideoCapture(path)
        self.frame_count = cap.get(cv2.CV_CAP_PROP_FRAME_COUNT)
        self.width = cap.get(cv2.CV_CAP_PROP_FRAME_WIDTH )
        self.height = cap.get(cv2.CV_CAP_PROP_FRAME_HEIGHT)
        self.pos_frame = 0

        self.lock = threading.Lock()

    def get_frame(self, frame_id):
        assert frame_id < self.frame_count, "Frame id out of bound %d" % self.frame_count
        with lock:
            if frame_id > self.pos_frame:
                # future frame: do sequential decode
                for _ in range(frame_id - self.pos_frame - 1):
                    self.cap.read()
            else:
                # past frame: rewind
                self.cap.set(cv2.CV_CAP_PROP_POS_FRAMES, frame_id)

            ret, frame = self.cap.read()

        self.pos_frame = frame_id
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def __del__(self):
        self.cap.release()