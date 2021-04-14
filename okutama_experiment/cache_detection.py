import gc
import hashlib
import logging
import itertools
import json
import operator
import os
from pathlib import Path
from typing import List

import cv2
import fire
from logzero import logger
import numpy as np
from numpy.linalg import norm

from rekall.bounds import Bounds3D
from rekall.bounds.utils import bounds_intersect, bounds_span
from rekall.predicates import (
    _area, _height, _iou, _width, and_pred, length_at_least, meets_before, iou_at_least, 
    or_pred, overlaps_before, starts
)

from stsearch.cvlib import *
from stsearch.interval import *
from stsearch.op import *
from stsearch.stdlib import centroid, same_time, average_space_span_time, tiou, tiou_at_least
from stsearch.utils import run_to_finish, VisualizeTrajectoryOnFrameGroup
from stsearch.videolib import *

logger.setLevel(logging.INFO)

DETECTION_SERVERS = [
    'agra.diamond.cs.cmu.edu',
    'briolette.diamond.cs.cmu.edu',
    'cullinan.diamond.cs.cmu.edu',
    'dresden.diamond.cs.cmu.edu',
    'indore.diamond.cs.cmu.edu',
    'kimberly.diamond.cs.cmu.edu',
    'patiala.diamond.cs.cmu.edu',
    'transvaal.diamond.cs.cmu.edu',
    'cloudlet031.elijah.cs.cmu.edu'
    ]
DETECTION_SERVERS = [f"{h}:{p}" for h,p in itertools.product(DETECTION_SERVERS, [5000, 5001])]

def query(path, session):

    import hashlib
    h = hashlib.md5()
    with open(path, 'rb') as f:
        h.update(f.read())
    file_digest = str(h.hexdigest())
    
    query_result = {}

    decoder = LocalVideoDecoder(path)
    frame_count, fps = decoder.frame_count, int(np.round(decoder.fps))
    query_result = {
        'fps': fps,
        'frame_count': frame_count,
        'width': decoder.raw_width,
        'height': decoder.raw_height,
        'file_digest': file_digest
    }
    del decoder

    all_frames = VideoToFrames(LocalVideoDecoder(path))()
    tiles = Tile(3, 3)(all_frames)

    tile_detections = Detection(server_list=DETECTION_SERVERS, parallel=len(DETECTION_SERVERS))(tiles)

    detections = []
    def map_fn(intrvl):
        D = intrvl.payload[DEFAULT_DETECTION_KEY]
        for class_name, score, (top, left, bottom, right) in \
            zip(D['detection_names'], D['detection_scores'], D['detection_boxes']):

            if score > 0.:
                # convert in-tile coords to in-frame coord
                x1 = intrvl['x1'] + left * intrvl.bounds.width()
                x2 = intrvl['x1'] + right * intrvl.bounds.width()
                y1 = intrvl['y1'] + top * intrvl.bounds.height()
                y2 = intrvl['y1'] + bottom * intrvl.bounds.height()

                detections.append({
                    'current_frame': intrvl['t1'],
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'class_name': class_name,
                    'score': score
                })

        return Interval(intrvl.bounds.copy())

    tile_detections = Map(map_fn)(tile_detections)
    output = tile_detections

    output_sub = output.subscribe()
    output.start_thread_recursive()

    _ = list(output_sub)

    query_result['detections'] = detections
    return query_result


def main(mode, path=None, cache_dir='okutama_cache'):
    assert mode in ('remote', 'local')

    if mode == 'remote':
        from pathlib import Path
        import pickle
        import time

        import pandas as pd 
        from stsearch.diamond_wrap.result_pb2 import STSearchResult
        from stsearch.diamond_wrap.utils import start_stsearch_by_script, OUTPUT_ATTR

        tic = time.time()

        results = start_stsearch_by_script(open(__file__, 'rb').read())

        for i, res in enumerate(results):
            # each `res` corresponds to results of a clip_id
            object_id = res['_ObjectID'].decode()
            clip_id = Path(object_id).stem
            print(f"=> Result {i}. Time {(time.time()-tic)/60:.1f} min. Clip {clip_id}")

            filter_result: STSearchResult = STSearchResult()
            filter_result.ParseFromString(res[OUTPUT_ATTR])
            query_result = pickle.loads(filter_result.query_result)
            query_result['clip_id'] = clip_id

            with open(Path(cache_dir)/f"{query_result['file_digest']}.json", 'wt') as f:
                json.dump(query_result, f, indent=2)

    elif mode == 'local':
        from pathlib import Path
        assert path is not None

        query_result = query(path, session=None)
        clip_id = Path(path).stem
        logger.info(query_result)

if __name__ == "__main__":
    fire.Fire(main)