import io
import random
import requests
from typing import Iterable, Optional

import cv2
import numpy as np
from logzero import logger

from rekall.bounds import Bounds3D
from stsearch.op import Filter, Graph, Map, Op
from stsearch.videolib import ImageInterval

DEFAULT_DETECTION_KEY = 'CVLIB_DETECTION_KEY'

class Detection(Graph):
    """Detection result is written to input interval *in-place*.

    """

    def __init__(
        self, 
        server='localhost', port=5000, 
        server_list: Optional[Iterable[str]]=None,
        result_key=DEFAULT_DETECTION_KEY):

        super().__init__()

        self.server_list = list(server_list or [f"{server}:{port}", ])
        self.result_key = result_key


    def call(self, instream):

        def map_fn(intrvl):
            assert isinstance(intrvl, ImageInterval)

            server = random.choice(self.server_list)
            detect_url = f"http://{server}/detect"

            r = requests.post(detect_url, files={'image': io.BytesIO(intrvl.jpeg)})
            assert r.ok
            result = r.json()
            if result['success']:
                intrvl.payload[self.result_key] = result
                # logger.debug(result)
            else:
                raise RuntimeError(str(result))
            
            return intrvl

        return Map(map_fn)(instream)

    
class DetectionVisualize(Graph):

    def __init__(self, targets, confidence=0.9, result_key=DEFAULT_DETECTION_KEY):
        super().__init__()

        assert iter(targets)
        self.targets = targets
        self.confidence = confidence
        self.result_key = result_key
        
        def map_fn(intrvl):
            try:
                detections = intrvl.payload[self.result_key]
            except KeyError:
                raise KeyError( f"Cannot find {self.result_key} in input payload. Did you run object detection on the input stream?")

            rgb = intrvl.rgb
            for box, score, class_name in zip(detections['detection_boxes'], detections['detection_scores'], detections['detection_names']):
                if score < self.confidence:
                    break
                for t in self.targets:
                    if t in class_name:
                        top, left, bottom, right = box  # TF return between 0~1
                        H, W = intrvl.rgb.shape[:2]
                        top, left, bottom, right = int(top*H), int(left*W), int(bottom*H), int(right*W) # to pixels
                        rgb = cv2.rectangle(rgb, (left, top), (right, bottom), (0, 255, 0), 3)
            
            intrvl.rgb = rgb
            return intrvl

        self.map_fn = map_fn

    def call(self, instream):
        return Map(self.map_fn)(instream)

class DetectionFilter(Graph):
    
    def __init__(self, targets, confidence=0.9, result_key=DEFAULT_DETECTION_KEY):
        super().__init__()

        assert iter(targets)
        self.targets = targets
        self.confidence = confidence
        self.result_key = result_key

    def call(self, instream):
        
        def pred_fn(intrvl):
            try:
                detections = intrvl.payload[self.result_key]
            except KeyError:
                raise KeyError( f"Cannot find {self.result_key} in input payload. Did you run object detection on the input stream?")

            for box, score, class_name in zip(detections['detection_boxes'], detections['detection_scores'], detections['detection_names']):
                if score < self.confidence:
                    return False
                for t in self.targets:
                    if t in class_name:
                        return True
            return False

        return Filter(pred_fn)(instream)


class DetectionFilterFlatten(Op):

    def __init__(self, targets, confidence=0.9, result_key=DEFAULT_DETECTION_KEY, name=None):
        super().__init__()
        assert iter(targets)
        self.targets = targets
        self.confidence = confidence
        self.result_key = result_key

    def call(self, instream):
        self.instream = instream

    def execute(self):
        while True:
            intrvl = self.instream.get()
            if intrvl is None:
                return False

            try:
                detections = intrvl.payload[self.result_key]
            except KeyError:
                raise KeyError( f"Cannot find {self.result_key} in input payload. Did you run object detection on the input stream?")

            has_result = False
            for box, score, class_name in zip(detections['detection_boxes'], detections['detection_scores'], detections['detection_names']):
                if score < self.confidence:
                    break
                for t in self.targets:
                    if t in class_name:
                        has_result = True
                        # create new patch
                        top, left, bottom, right = box  # TF return between 0~1
                        new_bounds = Bounds3D(
                            intrvl['t1'], intrvl['t2'],
                            intrvl['x1'] + intrvl.bounds.width() * left,
                            intrvl['x1'] + intrvl.bounds.width() * right,
                            intrvl['y1'] + intrvl.bounds.height() * top,
                            intrvl['y1'] + intrvl.bounds.height() * bottom
                        )
                        new_patch = ImageInterval(new_bounds, root=intrvl.root)
                        self.publish(new_patch)
            if has_result:
                return True
            else:
                continue


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
