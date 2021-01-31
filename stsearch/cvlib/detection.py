import hashlib
import io
import json
import operator
from pathlib import Path
import random
import requests
from typing import Iterable, Optional

import cv2
import numpy as np
from logzero import logger

from rekall.bounds import Bounds3D
from stsearch.op import Filter, Flatten, Graph, Map, Op
from stsearch.parallel import ParallelMap
from stsearch.videolib import ImageInterval

DEFAULT_DETECTION_KEY = 'CVLIB_DETECTION_KEY'

class Detection(Graph):
    """Detection result is written to input interval *in-place*.

    """

    def __init__(
        self, 
        server='localhost', port=5000, 
        server_list: Optional[Iterable[str]]=None,
        result_key=DEFAULT_DETECTION_KEY,
        parallel=1,
        name=None):

        super().__init__()

        self.server_list = list(server_list or [f"{server}:{port}", ])
        self.result_key = result_key
        self.name = name
        self.parallel = parallel


    def call(self, instream):
        name = self.name or f"{self.__class__.__name__}:{self.result_key}"

        def map_fn(intrvl):
            assert isinstance(intrvl, ImageInterval)

            max_try = 5
            while max_try > 0:
                try:
                    server = random.choice(self.server_list)
                    detect_url = f"http://{server}/detect"
                    r = requests.post(detect_url, files={'image': io.BytesIO(intrvl.jpeg)})
                    assert r.ok
                    break
                except Exception as e:
                    logger.exception(e)
                    if max_try > 0:
                        max_try -= 1
                    else:
                        raise

            result = r.json()
            if result['success']:
                intrvl.payload[self.result_key] = result
                # logger.debug(result)
            else:
                raise RuntimeError(str(result))
            
            return intrvl

        if self.parallel == 1:
            return Map(map_fn, name=name)(instream)
        else:
            return ParallelMap(map_fn, name=name, max_workers=self.parallel)(instream)

    
class DetectionVisualize(Graph):

    def __init__(self, targets, confidence=0.9, result_key=DEFAULT_DETECTION_KEY, color=(0, 255, 0)):
        super().__init__()

        assert iter(targets)
        self.targets = targets
        self.confidence = confidence
        self.result_key = result_key
        self.color = color
        
        def map_fn(intrvl):
            try:
                detections = intrvl.payload[self.result_key]
            except KeyError:
                raise KeyError( f"Cannot find {self.result_key} in input payload. Did you run object detection on the input stream?")

            rgb = np.copy(intrvl.rgb)
            for box, score, class_name in zip(detections['detection_boxes'], detections['detection_scores'], detections['detection_names']):
                if score < self.confidence:
                    break
                for t in self.targets:
                    if t in class_name:
                        top, left, bottom, right = box  # TF return between 0~1
                        H, W = intrvl.rgb.shape[:2]
                        top, left, bottom, right = int(top*H), int(left*W), int(bottom*H), int(right*W) # to pixels
                        rgb = cv2.rectangle(rgb, (left, top), (right, bottom), self.color, 3)
            
            new_intrvl = intrvl.copy()
            new_intrvl.rgb = rgb
            return new_intrvl

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


class DetectionFilterFlatten(Graph):

    def __init__(self, targets, confidence=0.9, result_key=DEFAULT_DETECTION_KEY, name=None):
        super().__init__()
        assert iter(targets)
        self.targets = targets
        self.confidence = confidence
        self.result_key = result_key
        self.name = name or self.__class__.__name__

    def call(self, instream):

        def flatten_fn(intrvl):
            rv = []
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
                        # the following arithmetic should be correct even if `intrvl` is not full frame.
                        new_bounds = Bounds3D(
                            intrvl['t1'], intrvl['t2'],
                            intrvl['x1'] + intrvl.bounds.width() * left,
                            intrvl['x1'] + intrvl.bounds.width() * right,
                            intrvl['y1'] + intrvl.bounds.height() * top,
                            intrvl['y1'] + intrvl.bounds.height() * bottom
                        )
                        new_patch = ImageInterval(new_bounds, root=intrvl.root)
                        rv.append(new_patch)
            rv.sort(key=lambda i: i.bounds)
            return rv

        return Flatten(flatten_fn, name=self.name)(instream)



class CachedVIRATDetection(Graph):
    def __init__(self, path, cache_dir="/root/cache", result_key=DEFAULT_DETECTION_KEY):

        # load cache file and sort by t1, so that we can direct access by indexing
        h = hashlib.md5()
        with open(path, 'rb') as f:
            h.update(f.read())
        digest = str(h.hexdigest())
        cache_path = str(Path(cache_dir) / (digest+'.json'))
        with open(cache_path, 'rt') as f:
            C = json.load(f)

        cached_detection = C['detection']    # list of dict
        cached_detection = sorted(cached_detection, key=operator.itemgetter('t1')) 
        self.C = cached_detection

        self.result_key = result_key

    def call(self, instream):

        def map_fn(intrvl):
            t1 = int(intrvl['t1'])
            intrvl.payload[self.result_key] = self.C[t1]['detection']
            return intrvl

        return Map(map_fn)(instream)
