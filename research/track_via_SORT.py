import os
from pathlib import Path

import cv2
from logzero import logger
import numpy as np

from rekall.bounds import Bounds3D
from rekall.predicates import _area, _height, _width, meets_before, iou_at_least, overlaps_before

from stsearch.cvlib import Detection, SORTTrackByDetection, get_boxes_from_detection
from stsearch.interval import *
from stsearch.op import *
from stsearch.utils import run_to_finish
from stsearch.videolib import *

from utils import VisualizeTrajectoryOnFrameGroup

INPUT_NAME = "example.mp4"
OUTPUT_DIR = Path(__file__).stem + "_output"


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fps = 15
    detect_every = 2
    
    all_frames = VideoToFrames(LocalVideoDecoder(INPUT_NAME))()
    sampled_frames = Slice(step=detect_every)(all_frames)
    detections = Detection('cloudlet031.elijah.cs.cmu.edu', 5002)(sampled_frames)

    short_trajectories = SORTTrackByDetection(
        get_boxes_fn=get_boxes_from_detection(['person'], 0.3),
        window=fps*30,
        trajectory_key='trajectory',
        max_age=7,
        iou_threshold=0.01
    )(detections)

    def trajectory_merge_predicate(i1, i2):
        return meets_before(3)(i1, i2) \
            and iou_at_least(0.3)(i1.payload['trajectory'][-1], i2.payload['trajectory'][0])

    def trajectory_payload_merge_op(p1, p2):
        print(f"Merging two trajectories of lengths {len(p1['trajectory'])} and {len(p2['trajectory'])}")
        return {'trajectory': p1['trajectory'] + p2['trajectory']}

    long_trajectories = Coalesce(
        predicate=trajectory_merge_predicate,
        bounds_merge_op=Bounds3D.span,
        payload_merge_op=trajectory_payload_merge_op,
        epsilon=1.1*detect_every
    )(short_trajectories)

    long_trajectories = Filter(
        pred_fn=lambda intrvl: intrvl.bounds.length() >= fps * 5
    )(long_trajectories)

    raw_fg = VideoCropFrameGroup(LRULocalVideoDecoder(INPUT_NAME), copy_payload=True)(long_trajectories)

    visualize_fg = VisualizeTrajectoryOnFrameGroup('trajectory')(raw_fg)

    output = visualize_fg
    output_sub = output.subscribe()
    output.start_thread_recursive()

    for k, intrvl in enumerate(output_sub):
        assert isinstance(intrvl, FrameGroupInterval)
        out_name = f"{OUTPUT_DIR}/{k}-{intrvl['t1']}-{intrvl['t2']}-{intrvl['x1']:.2f}-{intrvl['y1']:.2f}.mp4"
        intrvl.savevideo(out_name, fps=fps)
        logger.debug(f"saved {out_name}")

    logger.info(
        "You should find cropped .mp4 videos that 'bounds' a moving person."
        " We use trackers from OpenCV."
        " The trajectory is visualized."
    )