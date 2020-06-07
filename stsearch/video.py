import cv2
import numpy as np

from stsearch.interval import ImageInterval, RGB_KEY

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
    pass