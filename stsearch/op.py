from rekall.bounds import *

from stsearch.interval import Interval
from stsearch.invertal_stream import IntervalStream
from stsearch.video import *

class Op(object):
    
    def __init__(self):
        super().__init__()
        self.output = None

    def call(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        # FIXME how about Op that composes other ops?
        # traverse all args, check type, and subscribe to it 
        for a in list(args) + list(kwargs.values()):
            assert isinstance(a, IntervalStream), "Should only pass IntervalStream into call()"
            a.children.append(self)

        self.call(*args, **kwargs)
        # create an output stream
        self.output = IntervalStream(parent=self)
        return self.output

    def execute(self):
        raise NotImplementedError

    def publish(self, i):
        assert isinstance(i, Interval)
        assert self.output is not None, "call() has not been called"
        self.output.put(i)


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

class Slice(Op):

    def __init__(self, start=0, end=None, step=1):
        super().__init__()
        self.start = start
        self.end = end
        self.step = step
        self.ind = 0

    def call(self, instream):
        self.instream = instream

    def execute(self):
        while True:
            itvl = self.instream.get()
            if not itvl:
                return False
            if self.ind >= 0 and (not self.end or self.ind < self.end) and (self.ind - self.start) % self.step == 0:
                self.publish(itvl)
                return True
            elif self.end is not None and self.ind >= self.end:
                return False
            else:
                pass
            self.ind += 1

class Crop(Op):
    def __init__(self, x1=0., x2=1., y1=0., y2=1.):
        super().__init__()
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def call(self, instream):
        self.instream = instream

    def execute(self):
        ii = self.instream.get()
        assert isinstance(ii, ImageInterval), f"Expect ImageInterval. Got {type(ii)} "
        crop_ii = ImageInterval(
            bounds=Bounds3D(0, 1, self.x1, self.x2, self.y1, self.y2),
            root=ii)
        self.publish(crop_ii)
        return True
