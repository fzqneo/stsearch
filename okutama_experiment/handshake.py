import gc
import hashlib
import logging
import itertools
import json
import operator
import os
from pathlib import Path
import pickle
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

OKUTAMA_CACHE_DIR = "/root/okutama_cache/"
GET_MP4 = False


def query(path, session):
    cv2.setNumThreads(8)
    
    query_result = {}

    decoder = LocalVideoDecoder(path)
    frame_count, fps = decoder.frame_count, int(np.round(decoder.fps))
    query_result['metadata'] = {
        'fps': fps,
        'frame_count': frame_count,
        'width': decoder.raw_width,
        'height': decoder.raw_height,
    }
    query_result['results'] = list()
    del decoder

    detect_step = 30
    
    person_1 = DetectionFilterFlatten(['person'], 0.3)(CachedOkutamaDetection(path, OKUTAMA_CACHE_DIR)())
    person_2 = DetectionFilterFlatten(['person'], 0.3)(CachedOkutamaDetection(path, OKUTAMA_CACHE_DIR)())

    def is_handshake(i1: Interval, i2: Interval) -> bool:
        dx, dy = centroid(i1) - centroid(i2)
        return i1['t1'] == i2['t1'] and i1['x1'] < i2['x1'] and dx <= 1.5*i1.bounds.width() and dy <= 1.5*i1.bounds.height()

    def merge_op(i1, i2):
        new_bounds = i2.bounds.span(i1)
        new_bounds['t2'] = new_bounds['t2'] 
        return Interval(new_bounds)

    handshake_event = Join(
        predicate=is_handshake,
        merge_op=merge_op,
        window=detect_step
    )(person_1, person_2)

    # dedup and merge final results
    handshake_event = Coalesce(predicate=iou_at_least(.001),epsilon=detect_step)(handshake_event)

    if GET_MP4:
        vis_decoder = LRULocalVideoDecoder(path, cache_size=300)
        raw_fg = VideoCropFrameGroup(vis_decoder, copy_payload=True)(handshake_event)
        output = raw_fg
    else:
        output = handshake_event

    output_sub = output.subscribe()
    output.start_thread_recursive()

    for _, intrvl in enumerate(output_sub):
        if GET_MP4:
            query_result['results'].append((intrvl.bounds.copy(), intrvl.get_mp4()))
        else:
            query_result['results'].append((intrvl.bounds.copy(), b''))

        del intrvl
        gc.collect()

    return query_result


def main(mode, path=None, result_file="handshake_result.csv", get_mp4=False, mp4_dir="handshake_mp4"):
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

        save_results= []

        for i, res in enumerate(results):
            # each `res` corresponds to results of a clip_id
            object_id = res['_ObjectID'].decode()
            clip_id = Path(object_id).stem
            print(f"=> Result {i}. Time {(time.time()-tic)/60:.1f} min. Clip {clip_id}")

            filter_result: STSearchResult = STSearchResult()
            filter_result.ParseFromString(res[OUTPUT_ATTR])
            query_result = pickle.loads(filter_result.query_result)
            metadata = query_result['metadata']

            for seq, (b, mp4) in enumerate(query_result['results']):
                if len(mp4) > 0:
                    with open(f"{mp4_dir}/{clip_id}_{seq}_{b['t1']}_{b['t2']}.mp4", 'wb') as f:
                        f.write(mp4)
                        logger.info(f"saved {f.name}")

                save_results.append(
                    {
                        'clip_id': clip_id,
                        't1': b['t1'],
                        't2': b['t2'],
                        'x1': b['x1'],
                        'x2': b['x2'],
                        'y1': b['y1'],
                        'y2': b['y2'],
                        'result_size': len(mp4),
                        'frame_count': metadata['frame_count'],
                        'fps': metadata['fps'],
                        'width': metadata['width'],
                        'height': metadata['height'],
                    }
                )

            logger.info(f"# results = {len(query_result['results'])}")
            del query_result['results']
            logger.info(query_result)

            pd.DataFrame(save_results).to_csv(result_file)

    elif mode == 'local':
        from pathlib import Path
        assert path is not None
        if get_mp4:
            global GET_MP4
            GET_MP4 = True

        global OKUTAMA_CACHE_DIR
        OKUTAMA_CACHE_DIR = "/home/zf/video-analytics/stsearch/okutama_experiment/okutama_cache"
        query_result = query(path, session=None)
        clip_id = Path(path).stem
        for seq, (b, mp4) in enumerate(query_result['results']):
            if len(mp4) > 0:
                with open(f"{mp4_dir}/{clip_id}_{seq}_{b['t1']}_{b['t2']}.mp4", 'wb') as f:
                    f.write(mp4)
                    logger.info(f"saved {f.name}")

        logger.info(f"# results = {len(query_result['results'])}")
        del query_result['results']
        logger.info(query_result)


if __name__ == "__main__":
    fire.Fire(main)