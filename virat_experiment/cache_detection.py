import copy
import gc
import hashlib
import logging
import itertools
import os
from pathlib import Path
from typing import List

import cv2
from logzero import logger
import numpy as np
from numpy.linalg import norm

from rekall.bounds import Bounds3D
from rekall.bounds.utils import bounds_intersect, bounds_span
from rekall.predicates import _area, _height, _iou, _width, meets_before, iou_at_least, overlaps_before

from stsearch.cvlib import *
from stsearch.interval import *
from stsearch.op import *
from stsearch.stdlib import centroid, same_time
from stsearch.utils import run_to_finish, VisualizeTrajectoryOnFrameGroup
from stsearch.videolib import *


logger.setLevel(logging.INFO)

DIAMOND_SERVERS = [
    'agra.diamond.cs.cmu.edu',
    'briolette.diamond.cs.cmu.edu',
    'cullinan.diamond.cs.cmu.edu',
    'dresden.diamond.cs.cmu.edu',
    'indore.diamond.cs.cmu.edu',
    'kimberly.diamond.cs.cmu.edu',
    'patiala.diamond.cs.cmu.edu',
    'transvaal.diamond.cs.cmu.edu'
    ]
DETECTION_SERVERS = [f"{h}:{p}" for h,p in itertools.product(DIAMOND_SERVERS, [5000, 5001])]

# DETECTION_SERVERS = ["172.17.0.1:5000", "172.17.0.1:5001"]

# schemea:
# query_result = {
    # 'file_digest': ...,
    # 'height': ...,
    # ...,
    # 'detection': [
    #     {'t1': 1, 'detection': ...},
    #     {'t1': 2, 'detection': ...}
    # ]
# }

def query(path, session):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        h.update(f.read())
    file_digest = str(h.hexdigest())

    
    query_result = {
        'file_digest': file_digest,
        'detection': []
    }

    decoder = LocalVideoDecoder(path)
    frame_count, fps = decoder.frame_count, int(np.round(decoder.fps))
    query_result.update( {
        'fps': fps,
        'frame_count': frame_count,
        'width': decoder.raw_width,
        'height': decoder.raw_height,

    })

    del decoder
    
    all_frames = VideoToFrames(LocalVideoDecoder(path))()
    detections = Detection(server_list=DETECTION_SERVERS, parallel=2)(all_frames)

    output = detections

    output_sub = output.subscribe()
    output.start_thread_recursive()

    for _, intrvl in enumerate(output_sub):
        d = intrvl.payload[DEFAULT_DETECTION_KEY].copy()
        n = d['num_detections']
        d['detection_boxes'] = d['detection_boxes'][:n]
        d['detection_classes'] = d['detection_classes'][:n]
        d['detection_names'] = d['detection_names'][:n]
        d['detection_scores'] = d['detection_scores'][:n]

        query_result['detection'].append(
            {
                't1': int(intrvl['t1']),
                'detection': d
            }
        )

        del d
        del intrvl

    return query_result


if __name__ == "__main__":
    import json
    from pathlib import Path
    import pickle
    import time

    import pandas as pd 
    from stsearch.diamond_wrap.result_pb2 import STSearchResult
    from utils import start_stsearch_by_script, OUTPUT_ATTR

    RESULT_DIR = "cache"

    tic = time.time()

    results = start_stsearch_by_script(open(__file__, 'rb').read())

    for i, res in enumerate(results):
        logger.info(f"=> Result {i}. Time {(time.time()-tic)/60:.1f} min.")
        object_id = res['_ObjectID'].decode()
        clip_id = Path(object_id).stem

        filter_result: STSearchResult = STSearchResult()
        filter_result.ParseFromString(res[OUTPUT_ATTR])
        query_result = pickle.loads(filter_result.query_result)
        digest = query_result['file_digest']
        query_result['clip_id'] = clip_id


        with open(f"{RESULT_DIR}/{digest}.json", 'wt') as fout:
            json.dump(query_result, fout, indent=2)
