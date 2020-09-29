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
from rekall.predicates import _area, _height, _iou, _width, meets_before, iou_at_least, overlaps_before

from stsearch.cvlib import Detection, DetectionFilter, DetectionFilterFlatten, TrackFromBox
from stsearch.interval import *
from stsearch.op import *
from stsearch.stdlib import centroid, same_time
from stsearch.utils import run_to_finish
from stsearch.videolib import *

from utils import VisualizeTrajectoryOnFrameGroup

cv2.setNumThreads(4)
logger.setLevel(logging.INFO)

INPUT_NAME = "FifthCraig5_10m.mp4"
OUTPUT_DIR = Path(__file__).stem + "_output"

def traj_get_xy_array(L: List[Interval]) -> np.ndarray :
    return np.array([centroid(i) for i in L])

def linearize(pts, window=10):
    # simplify a trajectory (a list of x,y) into a line segment with two end points
    return np.mean(pts[:window, :], axis=0), np.mean(pts[-window:, :], axis=0)

def separate(a1, a2, b1, b2):
    # test if line (a1, a2) separate b1 and b2 into two sides
    # algorithm: use a1 as origin. dap = the normal vector of a
    # check whether b1 and b2 project to +/- sides of dap.
    da = a2 - a1
    dap = np.array([-da[1], da[0]])
    return np.dot(dap, b1-a1)*np.dot(dap, b2-a1) < 0

def seg_intersect(a1, a2, b1, b2):
    # note: test for line segment intersect, not line intersect
    return separate(a1, a2, b1, b2) and separate(b1, b2, a1, a2)  

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


def is_crosswalk(key1='traj_person', key2='traj_car', degree=50):

    def new_pred(i1: Interval, i2: Interval) -> bool:
        if not 0.1 <= _iou(i1, i2) <= 0.6:  # hack
            return False

        line1 = linearize(traj_get_xy_array(i1.payload[key1]))
        line2 = linearize(traj_get_xy_array(i2.payload[key2]))

        vec1 = line1[0] - line1[1]
        vec2 = line2[0] - line2[1]

        return seg_intersect(line1[0], line1[1], line2[0], line2[1]) \
            and np.arccos(abs(np.dot(vec1, vec2) / norm(vec1) / norm(vec2))) >= np.pi * degree/180

    return new_pred


def crosswalk_merge(key1='traj_person', key2='traj_car'):

    def new_interval_merge_op(i1: Interval, i2: Interval) -> Interval:
        # find the pairwise closest patches in the two trajectories
        v1, v2 = min(
            itertools.product(i1.payload[key1], i2.payload[key2]),
            key=lambda pair: np.linalg.norm(centroid(pair[0]) - centroid(pair[1]))
        )
        new_bounds = v1.bounds.span(v2.bounds)

        # span time, intersect space
        # new_bounds = i1.bounds.combine_per_axis(i2.bounds, bounds_span, bounds_intersect, bounds_intersect)
        
        new_payload = {
            key1: i1.payload[key1],
            key2: i2.payload[key2]
        }
        return Interval(new_bounds, new_payload)

    return new_interval_merge_op


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    decoder = LocalVideoDecoder(INPUT_NAME)
    frame_count, fps = decoder.frame_count, decoder.fps
    logger.info(f"Video info: frame_count {decoder.frame_count}, fps {decoder.fps}, raw_width {decoder.raw_width}, raw_height {decoder.raw_height}")
    del decoder

    detect_every = 8
    
    all_frames = VideoToFrames(LocalVideoDecoder(INPUT_NAME, resize=600))()
    sampled_frames = Slice(step=detect_every)(all_frames)
    detections = Detection('cloudlet031.elijah.cs.cmu.edu', 5000, parallel=2)(sampled_frames)
    crop_persons = DetectionFilterFlatten(['person'], 0.5)(detections)
    crop_cars = DetectionFilterFlatten(['car'], 0.5)(detections)
    
    track_person_trajectories = TrackFromBox(
        LRULocalVideoDecoder(INPUT_NAME, cache_size=450), 
        detect_every, 
        step=1,
        trajectory_key='traj_person',
        parallel_workers=12,
        name='track_person')(crop_persons)

    track_car_trajectories = TrackFromBox(
        LRULocalVideoDecoder(INPUT_NAME, cache_size=450), 
        detect_every,
        step=1,
        trajectory_key='traj_car',
        parallel_workers=24,
        name='track_car')(crop_cars)

    merged_person_trajectories = CoalesceByLast(
        predicate=traj_concatable(3, 0.3, 'traj_person'),
        bounds_merge_op=Bounds3D.span,
        payload_merge_op=traj_concat_payload('traj_person'),
        epsilon=1.1*detect_every
    )(track_person_trajectories)

    merged_car_trajectories = CoalesceByLast(
        predicate=traj_concatable(3, 0.3, 'traj_car'),
        bounds_merge_op=Bounds3D.span,
        payload_merge_op=traj_concat_payload('traj_car'),
        epsilon=1.1*detect_every
    )(track_car_trajectories)

    long_person_trajectories = Filter(
        pred_fn=lambda intrvl: intrvl.bounds.length() >= fps * 5
    )(merged_person_trajectories)

    long_car_trajectories = Filter(
        pred_fn=lambda intrvl: intrvl.bounds.length() >= fps * 5
    )(merged_car_trajectories)

    crosswalk_patches = JoinWithTimeWindow(
        predicate=is_crosswalk(),
        merge_op=crosswalk_merge('traj_person', 'traj_car'),
        window=int(15*60*2),  # 2 min
        name="join_crosswalk"
    )(long_person_trajectories, long_car_trajectories)

    vis_decoder = LRULocalVideoDecoder(INPUT_NAME, cache_size=900, resize=600)
    # raw_fg = VideoCropFrameGroup(vis_decoder, copy_payload=True, parallel=4)(crosswalk_patches)

    # # visualizing on many frames is very expensive, so we hack to shorten each fg to only 2 seconds
    # # comment this to get full visualization spanning both trajectories' times
    # def shorten(fg):
    #     fg.frames = fg.frames[:int(2*fps)]
    #     return fg
    # raw_fg = Map(map_fn=shorten)(raw_fg)

    # visualize_fg = VisualizeTrajectoryOnFrameGroup('traj_person', name="visualize-person-traj", parallel=4)(raw_fg)
    # visualize_fg = VisualizeTrajectoryOnFrameGroup('traj_car', name="visualize-car-traj", parallel=4)(visualize_fg)

    output = crosswalk_patches  # skip visualizing video for speed.
    # output = visualize_fg
    output_sub = output.subscribe()
    output.start_thread_recursive()

    reference_frame = cv2.cvtColor(LocalVideoDecoder(INPUT_NAME, resize=600).get_frame(10), cv2.COLOR_RGB2BGR)
    # for visualizing rect
    vis_rect = reference_frame.copy()
    # for visualizing voting/heatmap
    votes = np.zeros(reference_frame.shape[:2], dtype=np.int32)

    for k, intrvl in enumerate(output_sub):

        # assert isinstance(intrvl, FrameGroupInterval)
        # out_name = f"{OUTPUT_DIR}/{k}-{intrvl['t1']}-{intrvl['t2']}-{intrvl['x1']:.2f}-{intrvl['y1']:.2f}.mp4"
        # intrvl.savevideo(out_name, fps=fps)
        # logger.debug(f"saved {out_name}")

        # visualize box on vis_rect
        H, W = reference_frame.shape[:2]
        left, top = int(intrvl['x1'] * W), int(intrvl['y1'] * H)
        right, bottom = int(intrvl['x2'] * W), int(intrvl['y2'] * H)
        vis_rect = cv2.rectangle(vis_rect, (left, top), (right, bottom), (0, 255,0), 2)

        # add votes
        votes[top:bottom, left:right] += 1

    cv2.imwrite(f"{OUTPUT_DIR}/crosswalk.jpg", vis_rect)

    # visualize votes on heatmap
    hm = 155 * votes / np.max(votes)
    hm[np.nonzero(hm)] += 100   # clearly separate zero-vote
    hm = hm.astype(np.uint8)
    hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    vis_hm = cv2.addWeighted(hm, 0.5, reference_frame, 0.5, 0)
    cv2.imwrite(f"{OUTPUT_DIR}/crosswalk_hm.jpg", vis_hm)

    logger.info(vis_decoder.cache.cache_info())

    logger.info(
        "This example tries to find crosswalks by finding where human trajectories intersect with car trajectories."
    )
