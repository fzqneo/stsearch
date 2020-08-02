import os
import tempfile
import threading

import cv2
from logzero import logger
import numpy as np
import rekall

from stsearch.interval import Interval
from stsearch.op import *


# ROOT_KEY = '_root'  # a root image/frame to generate the current one
RGB_KEY = '_rgb'    # numpy array (H, W, 3)
JPEG_KEY = '_jpeg'
FRAMEGROUP_KEY = '_frames'  # a list of numpy array

class ImageInterval(Interval):

    def __init__(self, bounds, root=None):
        super().__init__(bounds)
        self.root = root or self

    @property
    def rgb(self):
        if RGB_KEY not in self.payload:
            assert self.root is not self, "Trying to make a crop with no root"
            rH, rW = self.root.rgb_height, self.root.rgb_width
            # crop
            self.payload[RGB_KEY] = self.root.rgb[
                int(rH*self['y1']):int(rH*self['y2']),
                int(rW*self['x1']):int(rW*self['x2']),
                :]
            # logger.debug(f"Origin shape: {self.root.rgb.shape}; crop shape {self.rgb.shape}")
        return self.payload[RGB_KEY]

    @rgb.setter
    def rgb(self, val):
        self.payload[RGB_KEY] = np.array(val)

    @property
    def rgb_height(self):
        return self.rgb.shape[0]

    @property
    def rgb_width(self):
        return self.rgb.shape[1]

    @property
    def jpeg(self):
        if JPEG_KEY not in self.payload:
            _, jpg_arr = cv2.imencode('.jpg', cv2.cvtColor(self.rgb, cv2.COLOR_RGB2BGR))
            self.payload[JPEG_KEY] = jpg_arr.tobytes()

        return self.payload[JPEG_KEY]

    @staticmethod
    def readfile(path):
        ii = ImageInterval(bounds=rekall.Bounds3D(0, 0))
        ii.rgb = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return ii

    def savefile(self, path):
        cv2.imwrite(path, cv2.cvtColor(self.rgb, cv2.COLOR_RGB2BGR))


class VideoFrameInterval(ImageInterval):

    def __init__(self, bounds, root_decoder=None):
        super().__init__(bounds, root=self)
        self.root_decoder = root_decoder
        assert isinstance(root_decoder, AbstractVideoDecoder)

    @property
    def rgb(self):
        if RGB_KEY not in self.payload:
            self.payload[RGB_KEY] = self.root_decoder.get_frame(self['t1'])
        return self.payload[RGB_KEY]

    @rgb.setter 
    def rgb(self, val):
        self.payload[RGB_KEY] = np.array(val)


class FrameGroupInterval(Interval):

    def __init__(self, bounds, payload=None):
        super().__init__(bounds, payload)

    @property
    def frames(self):
        assert FRAMEGROUP_KEY in self.payload, ".frames has not been set"
        return self.payload[FRAMEGROUP_KEY]

    @frames.setter
    def frames(self, val):
        assert iter(val)
        self.payload[FRAMEGROUP_KEY] = list(val)

    def savevideo(self, path, fps=30,
                  cv2_videowriter_fourcc=cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                  ):
        # .mp4 and .avi require different fourcc
        # we only do mp4 for now
        H, W = self.frames[0].shape[:2]
        logger.debug(f"VideoWrite fps={fps}, W={W}, H={H}")
        vw = cv2.VideoWriter(str(path), cv2_videowriter_fourcc, fps, (W,H))
        for im in self.frames:
            vw.write(im)
        vw.release()

    def get_mp4(self, *args, **kwargs):
        f = tempfile.NamedTemporaryFile('wb', suffix='.mp4', prefix='FrameGroupInterval', delete=False)
        f.close()
        self.savevideo(f.name)
        with open(f.name, 'rb') as f2:
            data = f2.read()
        os.unlink(f.name)
        return data

class AbstractVideoDecoder(object):
    def __init__(self):
        self.frame_count = None
        self.width = None
        self.height = None

    def get_frame(self, frame_id):
        raise NotImplementedError


class LocalVideoDecoder(AbstractVideoDecoder):
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
                bounds=rekall.Bounds3D(next_frame_id, next_frame_id+1),
                root_decoder=self.decoder)
            self.next_frame_id += 1
            self.publish(vfi)
            return True
        else:
            return False


class LocalVideoCropInterval(Graph):
    def __init__(self, path, copy_payload=True):
        self.decoder = LocalVideoDecoder(path)
        self.copy_payload = copy_payload

    def call(self, instream):
        def map_fn(intrvl):
            H, W = self.decoder.height, self.decoder.width
            if self.copy_payload:
                fg = FrameGroupInterval(intrvl.bounds.copy(), intrvl.payload)
            else:
                fg = FrameGroupInterval(intrvl.bounds.copy())
            # convert relative coordinate to absolute
            X1, X2 = int(intrvl['x1'] * W), int(intrvl['x2'] * W)
            Y1, Y2 = int(intrvl['y1'] * H), int(intrvl['y2'] * H)
            logger.debug(f"3D cropping: {intrvl['t1'], intrvl['t2'], X1, X2, Y1, Y2}")
            fg.frames = [
                self.decoder.get_frame(t1)[Y1:Y2, X1:X2, :]
                for t1 in range(intrvl['t1'], intrvl['t2'])
            ]
            return fg

        return Map(map_fn)(instream)


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
                bounds=rekall.Bounds3D(ii['t1'], ii['t2'], self.x1, self.x2, self.y1, self.y2),
                root=ii.root)
            return crop_ii

        return Map(map_fn)(instream)


# class VisualizeOnRoot(Graph):

#     def __init__(self):
#         super().__init__()

#     def call(self, instream):
#         def map_fn(ii):
#             assert isinstance(ii, ImageInterval)
#             new_ii = ImageInterval(ii.bounds.copy())
