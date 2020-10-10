import collections
import functools
import os
import tempfile
import typing
import threading

import cv2
from logzero import logger
import numpy as np

import rekall
import rekall.bounds

from stsearch.interval import Interval
from stsearch.op import *
from stsearch.parallel import ParallelMap


# ROOT_KEY = '_root'  # a root image/frame to generate the current one
RGB_KEY = '_rgb'    # numpy array (H, W, 3)
GRAY_KEY = '_gray'
JPEG_KEY = '_jpeg'
FRAMEGROUP_KEY = '_frames'  # a list of numpy array

class ImageInterval(Interval):

    def __init__(
        self, 
        bounds: rekall.bounds.Bounds, 
        payload: typing.Optional[dict] = None, 
        root: typing.Optional[Interval] = None):

        super().__init__(bounds, payload=payload)
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

    @rgb.deleter
    def rgb(self):
        if RGB_KEY in self.payload:
            self.payload.pop(RGB_KEY, None)

    @property
    def gray(self):
        # Can upgrade to functools.cached_property with Python>=3.8
        if GRAY_KEY not in self.payload:
            self.payload[GRAY_KEY] = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2GRAY)
        return self.payload[GRAY_KEY]

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

    def copy(self):
        return ImageInterval(
            self.bounds.copy(),
            payload=self.payload.copy(),
            root=self.root if self.root is not self else None) 

class VideoFrameInterval(ImageInterval):

    def __init__(self, bounds, payload=None, root_decoder=None):
        super().__init__(bounds, payload=payload, root=self)
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

    def copy(self):
        return VideoFrameInterval(
            bounds=self.bounds.copy(),
            payload=self.payload.copy(),
            root_decoder=self.root_decoder
        )


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
        # We can't use h264 encoding because of https://github.com/skvark/opencv-python/issues/100#issuecomment-394159998
        # the generated video may not play in most browsers.
        H, W = self.frames[0].shape[:2]
        logger.debug(f"VideoWrite fps={fps}, W={W}, H={H}")
        vw = cv2.VideoWriter(str(path), cv2_videowriter_fourcc, fps, (W,H))
        for j, im in enumerate(self.frames):
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
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

    def to_image_intervals(self) -> typing.List[ImageInterval]:
        """Convert the frames to a list of root-less ImageInterval"""

        rv = []
        for ts, frame in zip(range(int(self['t1']), int(self['t2'])), self.frames):
            new_bounds = self.bounds.copy()
            new_bounds['t1'], new_bounds['t2'] = ts, ts+1
            new_i = ImageInterval(new_bounds)
            new_i.rgb = frame
            rv.append(new_i)
        return rv


class AbstractVideoDecoder(object):
    def __init__(self, resize: typing.Union[None, int, typing.Sequence]=None):
        self.frame_count = None
        self.width = None
        self.height = None
        self.resize = resize

    def get_frame(self, frame_id):
        raw_frame = self.get_raw_frame(frame_id)
        if not self.resize:
            return raw_frame
        else:
            H, W = raw_frame.shape[:2]
            if isinstance(self.resize, (int, float)):
                sf = self.resize / max(H, W)
                new_W, new_H = int(sf*W), int(sf*H)
            else:
                new_W, new_H = self.resize

            return cv2.resize(raw_frame, (new_W, new_H))

    def get_raw_frame(self, frame_id):
        raise NotImplementedError

    def get_frame_interval(self, start_frame_id, end_frame_id, step=1):
        """Get a list of consecutive frames

        Args:
            start_frame_id (int): start inclusive
            end_frame_id (int): end exclusive

        Returns:
            Liast[np.ndarray]: [description]
        """
        return [self.get_frame(i) for i in range(start_frame_id, end_frame_id, step)]


class LocalVideoDecoder(AbstractVideoDecoder):
    def __init__(self, path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path = path
        cap = cv2.VideoCapture(path)
        self.cap = cap
        try:
            self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.raw_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH ))
            self.raw_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = cap.get(cv2.CAP_PROP_FPS)
        except AttributeError: # version difference
            self.frame_count = int(cap.get(cv2.CV_CAP_PROP_FRAME_COUNT))
            self.raw_width = int(cap.get(cv2.CV_CAP_PROP_FRAME_WIDTH ))
            self.raw_height = int(cap.get(cv2.CV_CAP_PROP_FRAME_HEIGHT))
            self.fps = cap.get(cv2.CV_CAP_PROP_FPS)
            
        self.pos_frame = 0  # frame id of the next read()

        self.lock = threading.Lock()

    def get_raw_frame(self, frame_id):
        assert frame_id < self.frame_count, f"Frame id {frame_id} out of bound {self.frame_count}" 
        # print(f"get raw frame {frame_id} {self.__class__.__name__}")
        with self.lock:
            if frame_id >= self.pos_frame:
                # future frame: do sequential decode
                for _ in range(frame_id - self.pos_frame):
                    self.cap.read()
            else:
                # past frame: rewind
                try:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                except AttributeError:
                    self.cap.set(cv2.CV_CAP_PROP_POS_FRAMES, frame_id)

            success, frame = self.cap.read()
            assert success and frame is not None, f"frame_id={frame_id}, frame_count={self.frame_count}, CV_CAP_PROP_POS_FRAMES={self.cap.get(cv2.CAP_PROP_POS_FRAMES)}"

        self.pos_frame = frame_id+1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def __del__(self):
        self.cap.release()

    def __str__(self):
        return f"{self.__class__.__name__}({self.path})"


class _LRUCache(collections.OrderedDict):
    # adapted from https://docs.python.org/3/library/collections.html#ordereddict-examples-and-recipes
    def __init__(self, maxsize, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.maxsize = maxsize
        self.hit = self.miss = 0
        
    # Not sure if we should move_to_end on GET
    def __getitem__(self, key):
        try:
            value = super().__getitem__(key)
            self.hit += 1
            return value
        except KeyError:
            self.miss += 1
            raise
        # self.move_to_end(key)

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            oldest = next(iter(self))
            del self[oldest]  

    def cache_info(self):
        return {'hit': self.hit, 'miss': self.miss}


class LRULocalVideoDecoder(LocalVideoDecoder):

    def __init__(self, path, cache_size=300, *args, **kwargs):

        super().__init__(path, *args, **kwargs)
        self.lock = threading.RLock()   # overwrite lock in super
        
        self.cache_size = cache_size
        self.cache = _LRUCache(cache_size)

    def get_frame(self, frame_id):
        with self.lock:
            try:
                return self.cache[frame_id]
            except KeyError:
                frame = super().get_frame(frame_id)
                self.cache[frame_id] = frame
                return frame

    def get_frame_interval(self, start_frame_id, end_frame_id, step=1):
        logger.debug(f"{self.__class__.__name__} get_frame_interval {start_frame_id} {end_frame_id} {step}")
        with self.lock:
            try:
                rv = [self.cache[i] for i in range(start_frame_id, end_frame_id, step)]
                return rv
            except KeyError:
                rv = super().get_frame_interval(start_frame_id, end_frame_id, step)
                self.cache.update(dict(zip(range(start_frame_id, end_frame_id, step), rv)))
                return rv


class VideoToFrames(Op):
    
    def __init__(self, decoder, name=None):
        super().__init__(name or f"{self.__class__.__name__}({str(decoder)})")
        assert isinstance(decoder, AbstractVideoDecoder)
        self.decoder = decoder
        self.next_frame_id = 0

    def call(self):
        # no input
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


class VideoCropFrameGroup(Graph):
    def __init__(self, decoder, copy_payload=True, parallel=1, name=None):
        assert isinstance(decoder, AbstractVideoDecoder)
        self.decoder = decoder
        self.copy_payload = copy_payload
        self.parallel = parallel
        self.name = name or f"{self.__class__.__name__}"

    def call(self, instream):
        def map_fn(intrvl):
            if self.copy_payload:
                fg = FrameGroupInterval(intrvl.bounds.copy(), intrvl.payload)
            else:
                fg = FrameGroupInterval(intrvl.bounds.copy())

            full_frames = self.decoder.get_frame_interval(intrvl['t1'], intrvl['t2'])
            H, W = full_frames[0].shape[:2]
            H, W = int(H), int(W)

            # convert relative coordinate to absolute
            X1, X2 = int(intrvl['x1'] * W), int(intrvl['x2'] * W)
            Y1, Y2 = int(intrvl['y1'] * H), int(intrvl['y2'] * H)
            # make sure they don't go out of bounds
            if not 0 <= X1 <= X2 <= W and 0 <= Y1 <= Y2 <= H:
                logger.warn(f"You're trying to crop out of bounds {[X1, X2, Y1, Y2]} from {[W, H]}")
            X1, X2 = max(X1, 0), min(X2, W)
            Y1, Y2 = max(Y1, 0), min(Y2, H)

            logger.debug(f"3D cropping: {intrvl['t1'], intrvl['t2'], X1, X2, Y1, Y2}")
            # perform spatial crop
            fg.frames =  [ frame[Y1:Y2, X1:X2, :] for frame in full_frames ]
            return fg

        return ParallelMap(map_fn, name=self.name, max_workers=self.parallel)(instream)


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

