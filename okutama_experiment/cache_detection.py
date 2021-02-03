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

    cv2.setNumThreads(8)
    
    query_result = {}

    decoder = LocalVideoDecoder(path)
    frame_count, fps = decoder.frame_count, int(np.round(decoder.fps))
    query_result['metadata'] = {
        'fps': fps,
        'frame_count': frame_count,
        'width': decoder.raw_width,
        'height': decoder.raw_height,
        'file_digest': file_digest
    }
    del decoder

    detect_step = 30
    
    all_frames = VideoToFrames(LocalVideoDecoder(path))()
    sampled_frames = Slice(step=detect_step)(all_frames)

    tiles = Tile(3, 3)(sampled_frames)

    detections = Detection(server_list=DETECTION_SERVERS, parallel=len(DETECTION_SERVERS))(tiles)
    cleaned_detections = Map(
        lambda i: Interval(i.bounds.copy(), {DEFAULT_DETECTION_KEY: i.payload[DEFAULT_DETECTION_KEY]})
    )(detections)

    output = cleaned_detections

    output_sub = output.subscribe()
    output.start_thread_recursive()

    query_result['detection'] = list(output_sub)
    return query_result


def main(mode, path=None, cache_dir='cache'):
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
            metadata = query_result['metadata']
            detection_result = query_result['detection']

            logger.info(metadata)
            with open(Path(cache_dir)/f"{metadata['file_digest']}.pkl", 'wb') as f:
                pickle.dump(detection_result, f)

    elif mode == 'local':
        from pathlib import Path
        assert path is not None

        query_result = query(path, session=None)
        clip_id = Path(path).stem
        logger.info(query_result)


if __name__ == "__main__":
    fire.Fire(main)