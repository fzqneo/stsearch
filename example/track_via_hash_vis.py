import os
from pathlib import Path

import cv2
from logzero import logger
import numpy as np

from rekall.bounds import Bounds3D
from rekall.predicates import _area, _height, _width, and_pred, before, iou_at_least

from stsearch.cvlib import Detection, DetectionFilter, DetectionFilterFlatten
from stsearch.interval import *
from stsearch.op import *
from stsearch.utils import run_to_finish
from stsearch.videolib import *

from utils import VisualizeTrajectoryOnFrameGroup

INPUT_NAME = "example.mp4"
OUTPUT_DIR = Path(__file__).stem + "_output"

def centroid(box):
    return np.array((0.5*(box['x1'] + box['x2']), 0.5*(box['y1'] + box['y2'])))

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fps = 15
    detect_every = 2
    
    all_frames = VideoToFrames(LocalVideoDecoder(INPUT_NAME))()
    sampled_frames = Slice(step=detect_every)(all_frames)
    detections = Detection('cloudlet031.elijah.cs.cmu.edu', 5000)(sampled_frames)
    crop_persons = DetectionFilterFlatten(['person'], 0.5)(detections)

    # Both phash and BMHash are okay. Marr hash not so good.
    hasher = cv2.img_hash.PHash_create()
    # hasher = cv2.img_hash.MarrHildrethHash_create()
    # hasher = cv2.img_hash.BlockMeanHash_create(cv2.img_hash.BLOCK_MEAN_HASH_MODE_0)

    def get_hash(intrvl):
        if 'img-hash' not in intrvl.payload:
            intrvl.payload['img-hash'] = hasher.compute(intrvl.rgb)
        return intrvl.payload['img-hash']

    def coalesce_predicate(i1, i2):
        hdiff = hasher.compare(get_hash(i1), get_hash(i2)) 
        L2 = np.linalg.norm(centroid(i1) - centroid(i2))
        return before(1)(i1, i2)  \
            and 0.5 <= _area(i1) / _area(i2) <= 2 \
            and hdiff <= 60 \
            and L2 < _height(i1)

    def coalesec_distance(i1, i2):
        hdiff = hasher.compare(get_hash(i1), get_hash(i2)) 
        L2 = np.linalg.norm(centroid(i1) - centroid(i2))
        return (hdiff, L2)

    def coalesce_interval_merge_op(i1, i2):
        new_bounds = i1.bounds.span(i2)
        new_payload = i1.payload.copy()

        # payload['trajectory'] will be a list of intervals
        if 'trajectory' not in new_payload:
            new_payload['trajectory'] = [i1, ]
        new_payload['trajectory'].append(i2)

        return Interval(new_bounds, new_payload)

    coalesced_persons = CoalesceByLast(
        predicate=coalesce_predicate,
        epsilon=detect_every*2,
        interval_merge_op=coalesce_interval_merge_op,
        distance=coalesec_distance)(crop_persons)

    long_coalesced_persons = Filter(
        pred_fn=lambda intrvl: intrvl.bounds.length() >= fps * 5
    )(coalesced_persons)

    raw_fg = VideoCropFrameGroup(LRULocalVideoDecoder(INPUT_NAME), copy_payload=True)(long_coalesced_persons)

    visualize_fg = VisualizeTrajectoryOnFrameGroup('trajectory')(raw_fg)

    output = visualize_fg
    output_sub = output.subscribe()
    output.start_thread_recursive()

    for k, intrvl in enumerate(output_sub):
        assert isinstance(intrvl, FrameGroupInterval)
        out_name = f"{OUTPUT_DIR}/{k}-{intrvl['t1']}-{intrvl['t2']}-{intrvl['x1']:.2f}-{intrvl['y1']:.2f}.mp4"
        intrvl.savevideo(out_name, fps=fps)
        logger.debug(f"saved {out_name}")

    logger.info('You should find cropped .mp4 videos that "bounds" a moving person. We use CoalesceByLast and image hash to do a form of tracking.')