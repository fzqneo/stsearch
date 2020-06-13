import cv2
import numpy as np

from stsearch.interval import ImageInterval, RGB_KEY
from stsearch.op import *
import threading

class VideoFrameInterval(ImageInterval):

    def __init__(self, bounds, root_decoder=None):
        super().__init__(bounds, root=None)
        self.root_decoder = root_decoder

    @property
    def rgb(self):
        if RGB_KEY not in self.payload:
            self.payload[RGB_KEY] = self.root_decoder.get_frame(self['t1'])
        return self.payload[RGB_KEY]

    @rgb.setter 
    def rgb(self, val):
        self.payload[RGB_KEY] = np.array(val)


class LocalVideoDecoder(object):
    def __init__(self, path):
        super().__init__()
        self.path = path
        cap = cv2.VideoCapture(path)
        self.cap = cap
        try:
            self.frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.width = cap.get(cv2.CAP_PROP_FRAME_WIDTH )
            self.height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        except AttributeError: # version difference
            self.frame_count = cap.get(cv2.CV_CAP_PROP_FRAME_COUNT)
            self.width = cap.get(cv2.CV_CAP_PROP_FRAME_WIDTH )
            self.height = cap.get(cv2.CV_CAP_PROP_FRAME_HEIGHT)
            
        self.pos_frame = 0

        self.lock = threading.Lock()

    def get_frame(self, frame_id):
        assert frame_id < self.frame_count, "Frame id out of bound %d" % self.frame_count
        with self.lock:
            if frame_id > self.pos_frame:
                # future frame: do sequential decode
                for _ in range(frame_id - self.pos_frame - 1):
                    self.cap.read()
            else:
                # past frame: rewind
                try:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                except AttributeError:
                    self.cap.set(cv2.CV_CAP_PROP_POS_FRAMES, frame_id)

            ret, frame = self.cap.read()

        self.pos_frame = frame_id
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def __del__(self):
        self.cap.release()


class LocalVideoToFrames(Op):
    
    def __init__(self, path):
        super().__init__()
        self.decoder = LocalVideoDecoder(path)
        self.next_frame_id = 0

    def call(self):
        # not input
        pass

    def execute(self):
        next_frame_id = self.next_frame_id
        if next_frame_id < self.decoder.frame_count:
            vfi = VideoFrameInterval(
                bounds=Bounds3D(next_frame_id, next_frame_id+1),
                root_decoder=self.decoder)
            self.next_frame_id += 1
            self.publish(vfi)
            return True
        else:
            return False


class Crop(Graph):
    def __init__(self, x1=0., x2=1., y1=0., y2=1.):
        super().__init__()
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def call(self, instream):
        def map_fn(ii):
            assert isinstance(ii, ImageInterval), f"Expect ImageInterval. Got {type(ii)} "
            crop_ii = ImageInterval(
                bounds=Bounds3D(ii['t1'], ii['t2'], self.x1, self.x2, self.y1, self.y2),
                root=ii)
            return crop_ii

        return Map(map_fn)(instream)

