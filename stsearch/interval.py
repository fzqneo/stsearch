import cv2
from logzero import logger
import numpy as np

import rekall
import rekall.bounds

ROOT_KEY = '_root'  # a root image/frame to generate the current one
RGB_KEY = '_rgb'    # numpy array (H, W, 3)

class Interval(rekall.Interval):
    def __init__(self, bounds):
        # we force payload to be a dict
        super().__init__(bounds, dict())
    
class ImageInterval(Interval):

    def __init__(self, bounds, root=None):
        super().__init__(bounds)
        assert root is None or isinstance(root, ImageInterval)
        self.payload[ROOT_KEY] = root

    @property
    def root(self):
        return self.payload[ROOT_KEY]

    @property
    def rgb(self):
        if RGB_KEY not in self.payload:
            assert self.root, "Trying to make a crop with no root"
            rH, rW = self.root.height, self.root.width
            # crop
            self.payload[RGB_KEY] = self.root.rgb[
                int(rH*self['y1']):int(rH*self['y2']),
                int(rW*self['x1']):int(rW*self['x2']),
                :]
            logger.debug(f"Origin shape: {self.root.rgb.shape}; crop shape {self.rgb.shape}")
        return self.payload[RGB_KEY]

    @rgb.setter
    def rgb(self, val):
        self.payload[RGB_KEY] = np.array(val)

    @property
    def height(self):
        return self.rgb.shape[0]

    @property
    def width(self):
        return self.rgb.shape[1]

    @staticmethod
    def readfile(path):
        ii = ImageInterval(bounds=rekall.Bounds3D(0, 0))
        ii.rgb = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return ii

    def savefile(self, path):
        cv2.imwrite(path, cv2.cvtColor(self.rgb, cv2.COLOR_RGB2BGR))
