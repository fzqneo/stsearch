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

from utils import VisualizeTrajectoryOnFrameGroup

cv2.setNumThreads(4)
logger.setLevel(logging.INFO)

INPUT_NAME = "find_getout_candidate/12-8450-9021-0.51-0.10.mp4"
# INPUT_NAME = "VIRAT_getin.mp4"
OUTPUT_DIR = Path(__file__).stem + "_output"

DETECTION_SERVERS = ['cloudlet031.elijah.cs.cmu.edu:5000', 'cloudlet031.elijah.cs.cmu.edu:5001']

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


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    decoder = LocalVideoDecoder(INPUT_NAME)
    frame_count, fps = decoder.frame_count, int(np.round(decoder.fps))
    logger.info(f"Video info: frame_count {decoder.frame_count}, fps {decoder.fps}, raw_width {decoder.raw_width}, raw_height {decoder.raw_height}")
    del decoder

    detect_step = 10
    
    all_frames = VideoToFrames(LocalVideoDecoder(INPUT_NAME))()
    sampled_frames = Slice(step=detect_step, end=int(5*60*fps))(all_frames)
    detections = Detection(server_list=DETECTION_SERVERS, parallel=4)(sampled_frames)

    crop_cars = DetectionFilterFlatten(['car'], 0.5)(detections)
    stopped_cars = Coalesce(
        predicate=iou_at_least(0.7),
        bounds_merge_op=Bounds3D.span,
        epsilon=detect_step*3
    )(crop_cars)

    stopped_cars = Log("stopped_cars")(stopped_cars)

    # further de-dup as detection boxes can be flashy
    stopped_cars = Coalesce(
        predicate=iou_at_least(0.5),
        bounds_merge_op=Bounds3D.span,
        epsilon=detect_step*3
    )(stopped_cars)
    stopped_cars = Log("dedup_stopped_cars")(stopped_cars)

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
    redetect_volumnes = Log("redetect_volumes")(redetect_volumnes)
    redetect_fg = VideoCropFrameGroup(LRULocalVideoDecoder(INPUT_NAME, cache_size=900), name="crop_redetect_volume")(redetect_volumnes)
    
    redetect_patches = Flatten(
        flatten_fn=lambda fg: fg.to_image_intervals()
    )(redetect_fg)
    redetect_detection = Detection(server_list=DETECTION_SERVERS, parallel=4)(
         Slice(step=detect_step)(redetect_patches)
    )
    redetect_person = DetectionFilterFlatten(['person'], 0.3)(redetect_detection)

    redetect_person = Log("redetect_person")(redetect_person)
    
    rekey = 'traj_person'

    short_person_trajectories = TrackFromBox(
        LRULocalVideoDecoder(INPUT_NAME, cache_size=900), 
        window=detect_step, 
        step=1,
        trajectory_key=rekey,
        parallel_workers=24,
        name='track_person')(redetect_person)

    short_person_trajectories = Log("short_person_trajectories")(short_person_trajectories)

    long_person_trajectories = Coalesce(
        predicate=traj_concatable(detect_step*2, 0.1, rekey),
        bounds_merge_op=Bounds3D.span,
        payload_merge_op=traj_concat_payload(rekey),
        distance=lambda i1, i2: _iou(i1.payload[rekey][-1], i2),
        epsilon=3*detect_step
    )(short_person_trajectories)

    # long_person_trajectories = Filter(
    #     lambda i: i.bounds.length() >= fps
    # )(long_person_trajectories)

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

    def merge_op_getout(i1, i2):
        new_bounds = i1.bounds.span(i2)
        new_payload = {rekey: i2.payload[rekey]}
        return Interval(new_bounds, new_payload)

    get_out = JoinWithTimeWindow(
        # lambda ic, ip: ic['t1'] < ip['t1'] < ic['t2'] \
            # and np.linalg.norm(centroid(ic) - centroid(ip.payload[rekey][0])) < _height(ic), 
        lambda ic, ip: ic['t1'] < ip['t1'] < ic['t2'] and _iou(ic, ip.payload[rekey][0]) > 0.01,
        merge_op=merge_op_getout
    )(stopped_cars, long_person_trajectories)


    if SAVE_VIDEO:
        vis_decoder = LRULocalVideoDecoder(INPUT_NAME, cache_size=900, resize=600)
        raw_fg = VideoCropFrameGroup(vis_decoder, copy_payload=True)(get_out)
        visualize_fg = VisualizeTrajectoryOnFrameGroup(rekey, name="visualize-person-traj")(raw_fg)
        output = visualize_fg

    else:
        output = redetect_volumnes

    output_sub = output.subscribe()
    output.start_thread_recursive()

    for k, intrvl in enumerate(output_sub):

        if SAVE_VIDEO:
            assert isinstance(intrvl, FrameGroupInterval)
            out_name = f"{OUTPUT_DIR}/{k}-{intrvl['t1']}-{intrvl['t2']}-{intrvl['x1']:.2f}-{intrvl['y1']:.2f}-{intrvl['x2']:.2f}-{intrvl['y2']:.2f}.mp4"
            intrvl.savevideo(out_name, fps=fps)
            logger.info(f"saved {out_name}")

