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

# INPUT_NAME = "VIRAT_getout_crop.mp4"
INPUT_NAME = "VIRAT_getin.mp4"

OUTPUT_DIR = Path(__file__).stem + "_output"

USE_OPTICAL=False
SAVE_VIDEO=True

def traj_get_xy_array(L: List[Interval]) -> np.ndarray :
    return np.array([centroid(i) for i in L])


def traj_concatable(epsilon, iou_thres, key='trajectory'):
    """Returns a predicate function that tests whether two trajectories
    are "closed" enough so that they can be concatenated.

    Args:
        epsilon ([type]): [description]
        iou_thres ([type]): [description]
        key (str, optional): [description]. Defaults to 'trajectory'.
    """

    def new_pred(i1: Interval, i2: Interval) -> bool:
        return meets_before(epsilon)(i1, i2) \
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
    detections = Detection('cloudlet031.elijah.cs.cmu.edu', 5000, parallel=4)(sampled_frames)

    
    if USE_OPTICAL:
        track_person_trajectories = TrackOpticalFlowFromBoxes(
            get_boxes_fn=get_boxes_from_detection(['person',], 0.3),
            step=1,
            decoder=LRULocalVideoDecoder(INPUT_NAME, cache_size=900),
            window=detect_step,
            trajectory_key='traj_person',
            parallel=8
        )(detections)

    else:
        # use expensive trackers
        crop_persons = DetectionFilterFlatten(['person'], 0.5)(detections)

        track_person_trajectories = TrackFromBox(
            LRULocalVideoDecoder(INPUT_NAME, cache_size=900), 
            detect_step, 
            step=2,
            trajectory_key='traj_person',
            parallel_workers=12,
            name='track_person')(crop_persons)

    crop_cars = DetectionFilterFlatten(['car'], 0.5)(detections)
    stopped_cars = Coalesce(
        predicate=iou_at_least(0.5),
        bounds_merge_op=Bounds3D.span,
        epsilon=fps*3
    )(crop_cars)

    stopped_cars = Filter(pred_fn=lambda i: i.bounds.length() >= 3*fps)(stopped_cars)

    merged_person_trajectories = Coalesce(
        predicate=traj_concatable(3, 0.1, 'traj_person'),
        bounds_merge_op=Bounds3D.span,
        payload_merge_op=traj_concat_payload('traj_person'),
        epsilon=10
    )(track_person_trajectories)

    long_person_trajectories = Filter(
        pred_fn=lambda intrvl: intrvl.bounds.length() >= 3 * fps
    )(merged_person_trajectories)

    # candidate starts from when the car stops to when the person emerge
    def merge_candidate(iperson: Interval, icar: Interval) -> Interval:
        carh, carw = _height(icar), _width(icar)
        new_bounds = Bounds3D(
            t1=int(max(0, icar['t1'] - fps*3)), 
            t2=int(max(0, iperson['t1'] + fps*3)),
            x1=max(0, icar['x1'] - carw),
            x2=min(1, icar['x2'] + carw),
            y1=max(0, icar['y1'] - carh),
            y2=min(1, icar['y2'] + carh)
        )

        new_payload = {
            'traj_person': iperson.payload['traj_person'],
            'stopped_car': icar
        }

        return Interval(new_bounds, new_payload)

    redetect_volumes = JoinWithTimeWindow(
        predicate = lambda ip, ic: ic['t1']  < ip['t1'] < ic['t2'] and _iou(ip.payload['traj_person'][0], ic) >= 0.01,
        merge_op=merge_candidate
        )(long_person_trajectories, stopped_cars)


    if SAVE_VIDEO:
        vis_decoder = LRULocalVideoDecoder(INPUT_NAME, cache_size=900, resize=600)
        raw_fg = VideoCropFrameGroup(vis_decoder, copy_payload=True, parallel=4)(redetect_volumes)

        # # visualizing on many frames is very expensive, so we hack to shorten each fg to only 2 seconds
        # # comment this to get full visualization spanning both trajectories' times
        # def shorten(fg):
        #     fg.frames = fg.frames[:int(2*fps)]
        #     return fg
        # raw_fg = Map(map_fn=shorten)(raw_fg)

        visualize_fg = VisualizeTrajectoryOnFrameGroup('traj_person', name="visualize-person-traj")(raw_fg)
        output = visualize_fg

    else:
        output = redetect_volumes

    output_sub = output.subscribe()
    output.start_thread_recursive()

    for k, intrvl in enumerate(output_sub):

        if SAVE_VIDEO:
            assert isinstance(intrvl, FrameGroupInterval)
            out_name = f"{OUTPUT_DIR}/{k}-{intrvl['t1']}-{intrvl['t2']}-{intrvl['x1']:.2f}-{intrvl['y1']:.2f}.mp4"
            intrvl.savevideo(out_name, fps=fps)
            logger.debug(f"saved {out_name}")

