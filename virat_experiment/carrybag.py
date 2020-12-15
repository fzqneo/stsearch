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

    detect_step = int(fps)

    all_frames = VideoToFrames(LocalVideoDecoder(path))()
    sampled_frames = Slice(step=detect_step)(all_frames)
    # detections = Detection(server_list=DETECTION_SERVERS, parallel=16)(sampled_frames)
    detections = CachedVIRATDetection(path)(sampled_frames)

    persons = DetectionFilterFlatten(['person'], 0.3)(detections)

    def dilate_op(i:ImageInterval) -> ImageInterval:
        w, h = _width(i), _height(i)
        new_bounds = Bounds3D(
            i['t1'], i['t2'],
            x1=max(0, i['x1']-w*3),
            x2=min(1, i['x2']+w*3),
            y1=max(0, i['y1']-h/2),
            y2=min(1, i['y2']+h/2)
        )
        return ImageInterval(new_bounds, None, root=i.root)

    persons = Map(dilate_op)(persons)
    persons_detect = Detection(server_list=DETECTION_SERVERS, parallel=8)(persons)
    persons_with_bag = DetectionFilter(['handbag', 'backpack','suitcase'], 0.1)(persons_detect)

    # dedup and merge final results
    merged_volumes = Coalesce(epsilon=detect_step*3)(persons_with_bag)
    merged_volumes = Filter(lambda i: i.bounds.length()>1*fps)(merged_volumes)
    vis_decoder = LRULocalVideoDecoder(path, cache_size=900)
    raw_fg = VideoCropFrameGroup(vis_decoder, copy_payload=True)(merged_volumes)

    output = raw_fg
    output_sub = output.subscribe()
    output.start_thread_recursive()

    for _, intrvl in enumerate(output_sub):
        # query_result['results'].append((intrvl.bounds, b''))
        query_result['results'].append((intrvl.bounds.copy(), intrvl.get_mp4()))
        del intrvl
        gc.collect()

    return query_result


if __name__ == "__main__":
    from pathlib import Path
    import pickle
    import time

    import pandas as pd 
    from stsearch.diamond_wrap.result_pb2 import STSearchResult
    from utils import start_stsearch_by_script, OUTPUT_ATTR

    tic = time.time()

    results = start_stsearch_by_script(open(__file__, 'rb').read())

    save_results= []

    for i, res in enumerate(results):
        # each `res` corresponds to results of a clip_id
        print(f"=> Result {i}. Time {(time.time()-tic)/60:.1f} min.")
        object_id = res['_ObjectID'].decode()
        clip_id = Path(object_id).stem

        filter_result: STSearchResult = STSearchResult()
        filter_result.ParseFromString(res[OUTPUT_ATTR])
        query_result = pickle.loads(filter_result.query_result)
        metadata = query_result['metadata']
        print(f"{clip_id}, {filter_result.stats}, {metadata}. #={len(query_result['results'])}")
        for b, mp4 in query_result['results']:
            # FIXME there can be duplicated names and files get overwritten
            open(f"carrybag_{clip_id}_{b['t1']}_{b['t2']}.mp4", 'wb').write(mp4)

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

        # save after getting each clip so that we don't lose all in case of failure
        pd.DataFrame(save_results).to_csv("carrybag.csv")

