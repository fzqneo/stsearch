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
    # 'briolette.diamond.cs.cmu.edu',
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

    tiles = Tile(2, 2)(sampled_frames)

    detections = Detection(server_list=DETECTION_SERVERS, parallel=8)(tiles)
    # frames_with_person = DetectionFilter(targets=['suitcase'], confidence=0.01)(detections)
    frames_visualized = DetectionVisualize(targets=['person'], confidence=0.3)(detections)
    frames_visualized = DetectionVisualize(targets=['suitcase'], confidence=0.01, color=(255,0,0))(frames_visualized)
    frames_visualized = Resize(size=800)(frames_visualized)

    output = frames_visualized

    output_sub = output.subscribe()
    output.start_thread_recursive()

    for _, intrvl in enumerate(output_sub):
        query_result['results'].append((intrvl.bounds.copy(), intrvl.jpeg))

        del intrvl
        gc.collect()

    return query_result


def main(mode, path=None, result_file="person.csv", jpeg_dir="person_jpeg"):
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

            for seq, (b, jpeg_blob) in enumerate(query_result['results']):
                if len(jpeg_blob) > 0:
                    with open(f"{jpeg_dir}/{clip_id}_{seq}_{b['t1']}_{b['t2']}.mp4", 'wb') as f:
                        f.write(jpeg_blob)
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
                        'result_size': len(jpeg_blob),
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
        query_result = query(path, session=None)
        clip_id = Path(path).stem
        for seq, (b, jpeg_blob) in enumerate(query_result['results']):
            if len(jpeg_blob) > 0:
                with open(f"{jpeg_dir}/{clip_id}_{seq}_{b['t1']}_{b['t2']}.jpg", 'wb') as f:
                    f.write(jpeg_blob)
                    logger.info(f"saved {f.name}")

        logger.info(f"# results = {len(query_result['results'])}")
        del query_result['results']
        logger.info(query_result)


if __name__ == "__main__":
    fire.Fire(main)