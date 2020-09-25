import logging
import os
from pathlib import Path

import cv2
from logzero import logger
import numpy as np

from rekall.bounds import Bounds3D
from rekall.predicates import _area, _height, _width, meets_before, iou_at_least, overlaps_before

from stsearch.cvlib import Detection, DetectionFilterFlatten, TrackFromBox
from stsearch.interval import *
from stsearch.op import *
from stsearch.stdlib import centroid, same_time
from stsearch.utils import run_to_finish
from stsearch.videolib import *

from utils import VisualizeTrajectoryOnFrameGroup


logger.setLevel(logging.INFO)
cv2.setNumThreads(4)

INPUT_NAME = "example.mp4"
OUTPUT_DIR = Path(__file__).stem + "_output"

def is_pair(corrcoef=.9, trajectory_key='trajectory'):

    def new_pred(i1: Interval, i2: Interval) -> bool:
        assert trajectory_key in i1.payload
        assert trajectory_key in i2.payload

        if not id(i1) < id(i2) \
            or not same_time(75)(i1, i2) \
            or not iou_at_least(0.5)(i1, i2):
            return False

        logger.debug("checking trajecory corr")

        def get_txy(i):
            # returns .shape=(n, 3). Each row is (t, x, y)
            return np.array([ [j['t1'],] + list(centroid(j)) for j in i.payload[trajectory_key]] )

        txy_1, txy_2 = get_txy(i1), get_txy(i2)

        ts = txy_1[:, 0]    # use 1's time as reference
        txy_2 = np.stack((ts, np.interp(ts, txy_2[:, 0], txy_2[:, 1]), np.interp(ts, txy_2[:, 0], txy_2[:, 2])), axis=1)
        # logger.debug(f"txy_1={txy_1}\ntxy_2={txy_2}")
        corr_x = np.corrcoef(txy_1[:, 1], txy_2[:, 1])[0 ,1]
        corr_y = np.corrcoef(txy_1[:, 2], txy_2[:, 2])[0, 1]
        logger.debug(f"corr_x={corr_x}, corr_y={corr_y}")
        return corr_x >= corrcoef and corr_y >= corrcoef

    return new_pred

def pair_merge_op(i1: Interval, i2: Interval) -> Interval:
    new_bounds = i1.bounds.span(i2.bounds)
    new_payload = {
        'trajectory_1': i1.payload['trajectory'],
        'trajectory_2': i2.payload['trajectory']
    }

    ret = Interval(new_bounds, new_payload)
    logger.debug(f"merged pair: {str(ret)}")
    return ret


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fps = 15
    detect_every = 8
    
    all_frames = VideoToFrames(LocalVideoDecoder(INPUT_NAME))()
    sampled_frames = Slice(step=detect_every)(all_frames)
    detections = Detection('cloudlet031.elijah.cs.cmu.edu', 5000)(sampled_frames)
    crop_persons = DetectionFilterFlatten(['person'], 0.5)(detections)

    track_trajectories = TrackFromBox(
        LRULocalVideoDecoder(INPUT_NAME, cache_size=300), 
        detect_every,
        name="track_person",
        parallel_workers=32
    )(crop_persons)

    def trajectory_merge_predicate(i1, i2):
        return meets_before(5)(i1, i2) \
            and iou_at_least(0.3)(i1.payload['trajectory'][-1], i2.payload['trajectory'][0])

    def trajectory_payload_merge_op(p1, p2):
        logger.debug(f"Merging two trajectories of lengths {len(p1['trajectory'])} and {len(p2['trajectory'])}")
        return {'trajectory': p1['trajectory'] + p2['trajectory']}

    coalesced_trajectories = Coalesce(
        predicate=trajectory_merge_predicate,
        bounds_merge_op=Bounds3D.span,
        payload_merge_op=trajectory_payload_merge_op,
        epsilon=3
    )(track_trajectories)

    long_coalesced_persons = Filter(
        pred_fn=lambda intrvl: intrvl.bounds.length() >= fps * 5
    )(coalesced_trajectories)

    pairs = JoinWithTimeWindow(
        predicate=is_pair(),
        merge_op=pair_merge_op,
        window=150
    )(long_coalesced_persons, long_coalesced_persons)

    raw_fg = VideoCropFrameGroup(LRULocalVideoDecoder(INPUT_NAME), copy_payload=True)(pairs)

    visualize_fg = VisualizeTrajectoryOnFrameGroup('trajectory_1')(raw_fg)
    visualize_fg = VisualizeTrajectoryOnFrameGroup('trajectory_2')(visualize_fg)

    output = visualize_fg
    output_sub = output.subscribe()
    output.start_thread_recursive()

    for k, intrvl in enumerate(output_sub):
        assert isinstance(intrvl, FrameGroupInterval)
        out_name = f"{OUTPUT_DIR}/{k}-{intrvl['t1']}-{intrvl['t2']}-{intrvl['x1']:.2f}-{intrvl['y1']:.2f}.mp4"
        intrvl.savevideo(out_name, fps=fps)
        logger.info(f"saved {out_name}")

    logger.info(
        "This example tries to find a pair of people walking together."
    )