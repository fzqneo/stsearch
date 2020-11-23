import gc
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
from rekall.predicates import _area, _height, _iou, _width, length_at_least, meets_before, iou_at_least, overlaps_before

from stsearch.cvlib import *
from stsearch.interval import *
from stsearch.op import *
from stsearch.stdlib import centroid, same_time
from stsearch.utils import run_to_finish
from stsearch.videolib import *

logger.setLevel(logging.INFO)

DETECTION_SERVERS = ["172.17.0.1:5000", "172.17.0.1:5001"]

def traj_concatable(epsilon, iou_thres, key='trajectory'):
    """Returns a predicate function that tests whether two trajectories
    are "closed" enough so that they can be concatenated.

    Args:
        epsilon ([type]): [description]
        iou_thres ([type]): [description]
        key (str, optional): [description]. Defaults to 'trajectory'.
    """

    def new_pred(i1: Interval, i2: Interval) -> bool:
        return i1['t1'] < i2['t1'] \
            and meets_before(epsilon)(i1, i2) \
            and iou_at_least(iou_thres)(i1.payload[key][-1], i2.payload[key][0])

    return new_pred

def traj_concat_payload(key):

    def new_payload_op(p1: dict, p2: dict) -> dict:
        logger.debug(f"Merging two trajectories of lengths {len(p1[key])} and {len(p2[key])}")
        return {key: p1[key] + p2[key]}

    return new_payload_op


def query(path, session):
    cv2.setNumThreads(8)
    
    query_result = {}

    decoder = LocalVideoDecoder(path)
    frame_count, fps = decoder.frame_count, int(np.round(decoder.fps))
    query_result['metadata'] = {
        'fps': fps,
        'frame_count': frame_count,
        'raw_w': decoder.raw_width,
        'raw_h': decoder.raw_height,
    }
    query_result['results'] = list()
    del decoder

    detect_step = int(fps)
    
    all_frames = VideoToFrames(LocalVideoDecoder(path))()
    sampled_frames = Slice(step=detect_step)(all_frames)
    detections = Detection(server_list=DETECTION_SERVERS, parallel=2)(sampled_frames)

    crop_cars = DetectionFilterFlatten(['car'], 0.5)(detections)
    stopped_cars = Coalesce(
        predicate=iou_at_least(0.7),
        bounds_merge_op=Bounds3D.span,
        epsilon=detect_step*3
    )(crop_cars)

    # further de-dup as detection boxes can be flashy
    stopped_cars = Coalesce(
        predicate=iou_at_least(0.5),
        bounds_merge_op=Bounds3D.span,
        epsilon=detect_step*3
    )(stopped_cars)

    stopped_cars = Filter(pred_fn=lambda i: i.bounds.length() >= 3*fps)(stopped_cars)

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

    redetect_volumnes = Map(dilate_car)(stopped_cars)

    redetect_fg = VideoCropFrameGroup(LRULocalVideoDecoder(path, cache_size=900), name="crop_redetect_volume")(redetect_volumnes)
    
    redetect_patches = Flatten(
        flatten_fn=lambda fg: fg.to_image_intervals()[::detect_step]
    )(redetect_fg)
    redetect_detection = Detection(server_list=DETECTION_SERVERS, parallel=2)(
         Slice(step=detect_step)(redetect_patches)
    )
    redetect_person = DetectionFilterFlatten(['person'], 0.3)(redetect_detection)

    rekey = 'traj_person'

    short_person_trajectories = TrackFromBox(
        LRULocalVideoDecoder(path, cache_size=900), 
        window=detect_step, 
        step=2,
        trajectory_key=rekey,
        parallel_workers=2,
        name='track_person')(redetect_person)

    long_person_trajectories = Coalesce(
        predicate=traj_concatable(detect_step*2, 0.1, rekey),
        bounds_merge_op=Bounds3D.span,
        payload_merge_op=traj_concat_payload(rekey),
        distance=lambda i1, i2: _iou(i1.payload[rekey][-1], i2),
        epsilon=3*detect_step
    )(short_person_trajectories)

    def interval_merge_op_1(i1, i2):
        new_bounds = i1.bounds.span(i2)
        new_payload = i1.payload.copy()

        if rekey not in new_payload:
            new_payload[rekey] = [i1, ]
        new_payload[rekey].append(i2)
        return Interval(new_bounds, new_payload)

    redetect_person_traj = CoalesceByLast(
        predicate=iou_at_least(0.3),
        interval_merge_op=interval_merge_op_1,
        epsilon=5
    )(redetect_person)
    redetect_person_traj = Filter(lambda i: i.bounds.length() > fps)(redetect_person_traj)

    def merge_op_getout(ic, ip):
        new_bounds = ic.bounds.span(ip)
        new_bounds['t1'] = max(0, ip['t1'] - fps)
        new_bounds['t2'] = min(frame_count, ip['t1'] + 10*fps)
        new_payload = {rekey: ip.payload[rekey]}
        return Interval(new_bounds, new_payload)

    get_out = JoinWithTimeWindow(
        # lambda ic, ip: ic['t1'] < ip['t1'] < ic['t2'] \
            # and np.linalg.norm(centroid(ic) - centroid(ip.payload[rekey][0])) < _height(ic), 
        lambda ic, ip: ic['t1'] < ip['t1'] < ic['t2'] and _iou(ic, ip.payload[rekey][0]) > 0.05,
        merge_op=merge_op_getout
    )(stopped_cars, long_person_trajectories)


    vis_decoder = LRULocalVideoDecoder(path, cache_size=900)
    raw_fg = VideoCropFrameGroup(vis_decoder, copy_payload=True)(get_out)
    output = raw_fg

    output_sub = output.subscribe()
    output.start_thread_recursive()

    for _, intrvl in enumerate(output_sub):
        query_result['results'].append((intrvl.bounds.copy(), intrvl.get_mp4()))
        del intrvl
        gc.collect()
        # query_result['results'].append((intrvl.bounds, b''))

    return query_result


if __name__ == "__main__":
    from pathlib import Path
    import pickle
    import time

    from stsearch.diamond_wrap.result_pb2 import STSearchResult
    from utils import start_stsearch_by_script, OUTPUT_ATTR

    tic = time.time()

    results = start_stsearch_by_script(open(__file__, 'rb').read())

    for i, res in enumerate(results):
        print(f"=> Result {i}. Time {(time.time()-tic)/60:.1f} min.")
        object_id = res['_ObjectID'].decode()
        clip_id = Path(object_id).stem

        filter_result: STSearchResult = STSearchResult()
        filter_result.ParseFromString(res[OUTPUT_ATTR])
        query_result = pickle.loads(filter_result.query_result)
        print(f"{clip_id}, {filter_result.stats}, {query_result['metadata']}. #={len(query_result['results'])}")
        for b, mp4 in query_result['results']:
            open(f"getoutcar_{clip_id}_{b['t1']}_{b['t2']}.mp4", 'wb').write(mp4)


