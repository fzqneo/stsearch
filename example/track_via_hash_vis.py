import os
from pathlib import Path

import cv2
from logzero import logger
import numpy as np

from rekall.bounds import Bounds3D
from rekall.predicates import _area, _height, _width, iou_at_least

from stsearch.cvlib import Detection, DetectionFilter, DetectionFilterFlatten
from stsearch.interval import *
from stsearch.op import *
from stsearch.utils import run_to_finish
from stsearch.videolib import *

INPUT_NAME = "example.mp4"
OUTPUT_DIR = Path(__file__).stem + "_output"

def centroid(box):
    return np.array((0.5*(box['x1'] + box['x2']), 0.5*(box['y1'] + box['y2'])))

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fps = 15
    detect_every = 3
    
    all_frames = LocalVideoToFrames(INPUT_NAME)()
    sampled_frames = Slice(step=detect_every)(all_frames)
    detections = Detection('cloudlet031.elijah.cs.cmu.edu', 5000)(sampled_frames)
    crop_persons = DetectionFilterFlatten(['person'], 0.5)(detections)

    # Both phash and BMHash are okay. Marr hash not so good.
    hasher = cv2.img_hash.PHash_create()
    # hasher = cv2.img_hash.MarrHildrethHash_create()
    # hasher = cv2.img_hash.BlockMeanHash_create(cv2.img_hash.BLOCK_MEAN_HASH_MODE_0)

    def hash_map_fn(intrvl):
        h = hasher.compute(intrvl.rgb)
        intrvl.payload['hash'] = h
        return intrvl

    persons_with_hash = Map(hash_map_fn, name="image-hash")(crop_persons)

    def coalesce_predicate(i1, i2):
        hdiff = hasher.compare(i1.payload['hash'], i2.payload['hash']) 
        L2 = np.linalg.norm(centroid(i1) - centroid(i2))
        return i1['t1'] < i2['t1']  \
            and 0.5 <= _area(i1) / _area(i2) <= 2 \
            and (hdiff <= 10 and L2 < 0.5  or 
                hdiff <= 60 and L2 < _height(i1) * 0.5)

    def track_interval_merge_op(i1, i2):
        new_bounds = i1.bounds.span(i2)
        new_payload = i1.payload.copy()

        # payload['trajectory'] will be a list of intervals
        if 'trajectory' not in new_payload:
            new_payload['trajectory'] = [i1, ]
        new_payload['trajectory'].append(i2)

        return Interval(new_bounds, new_payload)

    tracked_persons = CoalesceByLast(
        predicate=coalesce_predicate,
        epsilon=detect_every*2,
        interval_merge_op=track_interval_merge_op)(persons_with_hash)

    long_coalesced_persons = Filter(
        pred_fn=lambda intrvl: intrvl.bounds.length() >= fps * 5
    )(tracked_persons)

    raw_fg = LocalVideoCropFrameGroup(INPUT_NAME, copy_payload=True)(long_coalesced_persons)

    def visualize_map_fn(fg):
        pts = np.array([centroid(intrvl) for intrvl in fg.payload['trajectory']])
        assert pts.shape[1] == 2
        # this is tricky: adjust from relative coord in original frame to pixel coord in the crop
        pts =   (pts - [fg['x1'], fg['y1']]) / [_width(fg), _height(fg)] *  [fg.frames[0].shape[1], fg.frames[0].shape[0]]
        pts = pts.astype(np.int32)

        color = (0, 255, 0)
        thickness = 2
        is_closed = False

        new_fg = FrameGroupInterval(fg.bounds)
        new_frames = []

        for fid, frame in enumerate(fg.frames):
            frame = frame.copy()
            f1 = cv2.putText(frame, f"visualize-{fid}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            # somehow polylines doesn't work
            # f1 = cv2.polylines(f1, pts, is_closed, color, thickness)
            for j, (p1, p2) in enumerate(zip(pts[:-1], pts[1:])):
                print("Drawing line", p1, p2)
                f1 = cv2.line(f1, tuple(p1), tuple(p2), color, thickness)
                f1 = cv2.putText(f1, str(j), tuple(p1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1, cv2.LINE_AA)
            new_frames.append(f1)

        new_fg.frames = new_frames
        return new_fg

    visualization = Map(map_fn=visualize_map_fn)(raw_fg)

    output = visualization
    output_sub = output.subscribe()
    output.start_thread_recursive()

    for k, intrvl in enumerate(output_sub):
        assert isinstance(intrvl, FrameGroupInterval)
        out_name = f"{OUTPUT_DIR}/{k}-{intrvl['t1']}-{intrvl['t2']}-{intrvl['x1']:.2f}-{intrvl['y1']:.2f}.mp4"
        intrvl.savevideo(out_name, fps=fps)
        logger.debug(f"saved {out_name}")

    logger.info('You should find cropped .mp4 videos that "bounds" a moving person. We use CoalesceByLast and image hash to do a form of tracking.')