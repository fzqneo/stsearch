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
from rekall.predicates import (
    _area, _height, _iou, _width, and_pred, length_at_least, meets_before, iou_at_least, 
    or_pred, overlaps_before, starts
)

from stsearch.cvlib import *
from stsearch.interval import *
from stsearch.op import *
from stsearch.stdlib import centroid, same_time, average_space_span_time, tiou, tiou_at_least
from stsearch.utils import run_to_finish
from stsearch.videolib import *

from utils import VisualizeTrajectoryOnFrameGroup

cv2.setNumThreads(4)
logger.setLevel(logging.DEBUG)

# INPUT_NAME = "find_getout_candidate/12-8450-9021-0.51-0.10.mp4"
INPUT_NAME = "VIRAT_getin.mp4"
OUTPUT_DIR = Path(__file__).stem + "_output"

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

# DETECTION_SERVERS = ['cloudlet031.elijah.cs.cmu.edu:5000', 'cloudlet031.elijah.cs.cmu.edu:5001']

SAVE_VIDEO=True

class Log(Graph):

    def __init__(self, tag):
        super().__init__()
        self.tag = tag

    def call(self, instream):
        def map_fn(i):
            logger.info(f"{self.tag} {i.bounds}")
            return i
        return Map(map_fn)(instream)



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
    # detections = Detection(server_list=DETECTION_SERVERS, parallel=16)(sampled_frames)
    detections = CachedVIRATDetection(path, cache_dir="../virat_experiment/cache")(sampled_frames)

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
        payload_merge_op=lambda p1, p2: {}, # drop the rgb
        epsilon=detect_step*3
    )(stopped_cars)

    stopped_cars = Filter(pred_fn=lambda i: i.bounds.length() >= 3*fps)(stopped_cars)

    # buffer all stopped cars
    stopped_cars_sub = stopped_cars.subscribe()
    stopped_cars.start_thread_recursive()
    buffered_stopped_cars = list(stopped_cars_sub)

    logger.info(f"Find {len(buffered_stopped_cars)} stopped cars. {list(c.bounds for c in buffered_stopped_cars)}")
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

    redetect_fg = VideoCropFrameGroup(LRULocalVideoDecoder(path, cache_size=900), name="crop_redetect_volume")(redetect_bounds)
    
    redetect_patches = Flatten(
        flatten_fn=lambda fg: fg.to_image_intervals()
    )(redetect_fg)

    # redetect_patches = Log("redetect_patches")(redetect_patches)

    # we already sample when generating `redetect_bounds`. Don't sample again here.
    redetect_detection = Detection(server_list=DETECTION_SERVERS, parallel=16)(redetect_patches) 
    redetect_person = DetectionFilterFlatten(['person'], 0.3)(redetect_detection)

    # redetect_person = Log("redetect_person")(redetect_person)

    rekey = 'traj_person'

    short_person_trajectories = TrackFromBox(
        LRULocalVideoDecoder(path, cache_size=900), 
        window=detect_step, 
        step=2,
        trajectory_key=rekey,
        bidirectional=True,
        parallel_workers=2,
        name='track_person')(redetect_person)

    long_person_trajectories = Coalesce(
        predicate=traj_concatable(3*detect_step, 0.1, rekey),
        bounds_merge_op=Bounds3D.span,
        payload_merge_op=traj_concat_payload(rekey),
        # distance=lambda i1, i2: _iou(i1.payload[rekey][-1], i2.payload[rekey][0]),
        epsilon=10*detect_step
    )(short_person_trajectories)

    def merge_op_getout(ic, ip):
        new_bounds = ic.bounds.span(ip)
        # new_bounds['t1'] = max(0, ip['t1'] - 10*fps) # wind back 3 seconds
        # new_bounds['t2'] = min(frame_count, ip['t1'] + 10*fps)
        new_payload = {rekey: ip.payload[rekey]}
        return Interval(new_bounds, new_payload)

    get_out = JoinWithTimeWindow(
        lambda ic, ip: ic['t1'] < ip['t1'] < ic['t2'] and _iou(ic, ip.payload[rekey][0]) > 0.05,
        merge_op=merge_op_getout
    )(FromIterable(buffered_stopped_cars)(), long_person_trajectories)

    # dedup final results
    get_out = Coalesce(
        predicate=and_pred(iou_at_least(0.5), or_pred(during_inv(), tiou_at_least(0.5))),
        payload_merge_op=lambda p1,p2: p1
    )(get_out)

    vis_decoder = LRULocalVideoDecoder(path, cache_size=900)
    raw_fg = VideoCropFrameGroup(vis_decoder, copy_payload=True)(get_out)
    visualize_fg = VisualizeTrajectoryOnFrameGroup(rekey, name="visualize-person-traj")(raw_fg)
    output = visualize_fg

    output_sub = output.subscribe()
    output.start_thread_recursive()

    for k, intrvl in enumerate(output_sub):
        assert isinstance(intrvl, FrameGroupInterval)
        out_name = f"{OUTPUT_DIR}/{k}-{intrvl['t1']}-{intrvl['t2']}-{intrvl['x1']:.2f}-{intrvl['y1']:.2f}-{intrvl['x2']:.2f}-{intrvl['y2']:.2f}.mp4"
        intrvl.savevideo(out_name, fps=fps)
        logger.info(f"saved {out_name}")



if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    query(INPUT_NAME, None)
