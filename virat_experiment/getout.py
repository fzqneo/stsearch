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

GET_MP4 = False
VIRAT_CACHE_DIR = "/root/cache/"
        
def traj_concatable(epsilon, iou_thres, key='trajectory'):
    """Returns a predicate function that tests whether two trajectories
    are "closed" enough so that they can be concatenated.
    """

    def new_pred(i1: Interval, i2: Interval) -> bool:
        logger.debug(f"checking two traj: {(i1['t1'], i1['t2'])}, {(i2['t1'], i2['t2'])}")
        if not (i1['t1'] < i2['t1'] and i1['t2'] < i2['t2'] and abs(i1['t2']-i2['t1']) <= epsilon):
            logger.debug("failed temporal condition")
            return False

        tr1 = i1.payload[key]
        tr2 = [j for j in i2.payload[key] if j['t1'] >= tr1[-1]['t1']]   # clamp the overlapping part
        N = 3
        ret = _iou(average_space_span_time(tr1[-N:]), average_space_span_time(tr2[:N])) > iou_thres
        if not ret:
            logger.debug("failed spatial conditoin")
            logger.debug("i1.traj="+'\n'.join([str(i.bounds) for i in i1.payload[key]]))
            logger.debug("i2.traj="+'\n'.join([str(i.bounds) for i in i2.payload[key]]))
        return ret

    return new_pred

def traj_concat_payload(key):

    def new_payload_op(p1: dict, p2: dict) -> dict:
        tr1 = p1[key]
        tr2 = [j for j in p2[key] if j['t1'] > tr1[-1]['t1']]   # clamp the overlapping part
        return {key: tr1+tr2}
    return new_payload_op


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

    rekey = 'traj_person'

    short_person_trajectories = TrackFromBox(
        LRULocalVideoDecoder(path, cache_size=300), 
        window=detect_step, 
        step=2,
        trajectory_key=rekey,
        bidirectional=True,
        parallel_workers=2,
        name='track_person')(redetect_person)

    short_person_trajectories = Sort(2*detect_step)(short_person_trajectories)

    long_person_trajectories = Coalesce(
        predicate=traj_concatable(3*detect_step, 0.1, rekey),
        bounds_merge_op=Bounds3D.span,
        payload_merge_op=traj_concat_payload(rekey),
        epsilon=10*detect_step
    )(short_person_trajectories)

    long_person_trajectories = Filter(
        lambda i: i.bounds.length()>3*fps)(long_person_trajectories)

    def merge_op_getout(ic, ip):
        new_bounds = ic.bounds.span(ip)
        new_bounds['t1'] = max(0, ip['t1'] - 3*fps) # wind back 3 seconds
        new_bounds['t2'] = min(frame_count, ip['t1'] + fps)
        new_payload = {rekey: ip.payload[rekey]}
        return Interval(new_bounds, new_payload)

    get_out = JoinWithTimeWindow(
        lambda ic, ip: ic['t1'] < ip['t1'] < ic['t2'] and _iou(ic, ip.payload[rekey][0]) > 0.05,
        merge_op=merge_op_getout,
        window=5*60*fps
    )(FromIterable(buffered_stopped_cars)(), long_person_trajectories)

    # dedup and merge final results
    get_out = Coalesce(predicate=overlaps())(get_out)

    if GET_MP4:
        vis_decoder = LRULocalVideoDecoder(path, cache_size=900)
        raw_fg = VideoCropFrameGroup(vis_decoder, copy_payload=True)(get_out)
        visualize_fg = VisualizeTrajectoryOnFrameGroup(rekey, name="visualize-person-traj")(raw_fg)
        output = visualize_fg
    else:
        output = get_out

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


def main(mode, path=None, result_file="getout_result.csv", get_mp4=True, mp4_dir="getout_mp4"):
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

        del query_result['results']
        logger.info(query_result)
        logger.info(f"# results = {seq}")


if __name__ == "__main__":
    fire.Fire(main)