from typing import Iterable, Optional

import cv2
import numpy as np
from logzero import logger

from rekall.bounds import Bounds3D
from stsearch.op import Filter, Graph, Map, Op
from stsearch.videolib import ImageInterval

class ColorFilter(Graph):

    def __init__(self, rgb_lo=(0, 0, 0), rgb_hi=(255, 255, 255), count=None, ratio=None):
        super().__init__()
        assert (count is not None and count>=0) or (ratio is not None and 0. <= ratio <= 1.)
        self.rgb_lo = np.array(rgb_lo).astype(np.uint8)
        self.rgb_hi = np.array(rgb_hi).astype(np.uint8)
        self.count = count
        self.ratio = ratio

    def call(self, instream):
        def filter_fn(i):
            mask = cv2.inRange(i.rgb, self.rgb_lo, self.rgb_hi)
            if self.count:
                return np.count_nonzero(mask) >= self.count
            else:
                return np.count_nonzero(mask) / float(np.size(mask)) >= self.ratio

        return Filter(filter_fn)(instream)        
