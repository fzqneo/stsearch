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

GET_MP4 = True
VIRAT_CACHE_DIR = "/root/cache/"


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

    # detections = Detection(server_list=DETECTION_SERVERS, parallel=2)(sampled_frames)
    detections = CachedVIRATDetection(path, cache_dir=VIRAT_CACHE_DIR)(sampled_frames)

    crop_cars = DetectionFilterFlatten(['car', 'truck', 'bus'], 0.3)(detections)
    stopped_cars = Coalesce(
        predicate=iou_at_least(0.7),
        bounds_merge_op=Bounds3D.span,
        epsilon=detect_step*3
    )(crop_cars)

    # further de-dup as detection boxes can be flashy
    stopped_cars = Coalesce(
        predicate=iou_at_least(0.5),
        bounds_merge_op=Bounds3D.span,
        payload_merge_op=lambda p1, p2: {}, # drop the rgb
        epsilon=detect_step*3
    )(stopped_cars)

    stopped_cars = Filter(pred_fn=lambda i: i.bounds.length() >= 3*fps)(stopped_cars)

    # buffer all stopped cars
    stopped_cars_sub = stopped_cars.subscribe()
    stopped_cars.start_thread_recursive()
    buffered_stopped_cars = list(stopped_cars_sub)

    logger.info(f"Find {len(buffered_stopped_cars)} stopped cars.")
    query_result['stopped_cars'] = len(buffered_stopped_cars)

    # Stage 2: reprocess buffered stopped cars
    gc.collect()
    stopped_cars_1 = FromIterable(buffered_stopped_cars)()
    
    def dilate_car(icar: Interval) -> Interval:
        carh, carw = _height(icar), _width(icar)
        new_bounds = Bounds3D(
            t1=int(max(0, icar['t1']-fps)), 
            t2=int(min(frame_count, icar['t2']+fps)),
            x1=max(0, icar['x1'] - carw),
            x2=min(1, icar['x2'] + carw),
            y1=max(0, icar['y1'] - carh),
            y2=min(1, icar['y2'] + carh)
        )
        return Interval(new_bounds)

    dialted_stopped_cars = Map(dilate_car)(stopped_cars_1)

    # sample single-frame bounds from redetect_volumnes for object detection
    redetect_bounds = Flatten(
        flatten_fn=lambda i: \
            [
                Interval(Bounds3D(t1, t1+1, i['x1'], i['x2'], i['y1'], i['y2'])) \
                for t1 in range(int(i['t1']), int(i['t2']), detect_step) 
            ]
    )(dialted_stopped_cars)
    redetect_bounds = Sort(window=frame_count)(redetect_bounds)

    redetect_fg = VideoCropFrameGroup(LRULocalVideoDecoder(path, cache_size=300), name="crop_redetect_volume")(redetect_bounds)
    
    redetect_patches = Flatten(
        flatten_fn=lambda fg: fg.to_image_intervals()
    )(redetect_fg)
    # Don't sample again here
    redetect_detection = Detection(server_list=DETECTION_SERVERS, parallel=len(DETECTION_SERVERS))(redetect_patches)
    redetect_person = DetectionFilterFlatten(['person'], 0.3)(redetect_detection)

    stationary_person = Coalesce(predicate=iou_at_least(0.1), epsilon=2*detect_step)(redetect_person)
    stationary_person = Filter(pred_fn=lambda i: i.bounds.length()>=2*fps)(stationary_person)

    def loading_merge_op(i_person, i_vehicle):
        new_bounds = i_vehicle.bounds.span(i_person)
        new_bounds['t1'] = i_person['t1'] 
        new_bounds['t2'] = i_person['t2']
        return Interval(new_bounds)

    loading_event = Join(
        predicate=and_pred(iou_at_least(0.1), during()),
        merge_op=loading_merge_op
    )(stationary_person, FromIterable(buffered_stopped_cars)())

    # dedup and merge final results
    loading_event = Coalesce(predicate=overlaps())(loading_event)

    if GET_MP4:
        vis_decoder = LRULocalVideoDecoder(path, cache_size=300)
        raw_fg = VideoCropFrameGroup(vis_decoder, copy_payload=True)(loading_event)
        output = raw_fg
    else:
        output = loading_event

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


def main(mode, path=None, result_file="loading_result.csv", get_mp4=True, mp4_dir="loading_mp4"):
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
        global VIRAT_CACHE_DIR
        VIRAT_CACHE_DIR = "/home/zf/video-analytics/stsearch/virat_experiment/cache"
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