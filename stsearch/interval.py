import numpy as np

import rekall

ROOT_KEY = '_root'  # a root image/frame to generate the current one
RGB_KEY = '_rgb'    # numpy array (H, W, 3)

class Interval(rekall.Interval):
    def __init__(self, bounds, payload=dict()):
        super().__init__(bounds, payload=payload)
    

class ImageInterval(Interval):

    def __init__(self, bounds, payload=dict(), root=None):
        super().__init__(bounds, payload=payload)
        assert root is None or isinstance(root, Interval)
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
        return self.payload[RGB_KEY]

    @rgb.setter
    def rgb(self, val):
        self.payload[RGB_KEY] = np.array(val)

    