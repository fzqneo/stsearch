import os
from pathlib import Path

import cv2
from logzero import logger
import numpy as np

from rekall.bounds import Bounds3D
from rekall.predicates import _area, _height, _width, meets_before, iou_at_least, overlaps_before

from stsearch.cvlib import Detection, DetectionFilter, DetectionFilterFlatten
from stsearch.interval import *
from stsearch.op import *
from stsearch.utils import run_to_finish
from stsearch.videolib import *

INPUT_NAME = "example.mp4"
OUTPUT_DIR = Path(__file__).stem + "_output"

def centroid(box):
    return np.array((0.5*(box['x1'] + box['x2']), 0.5*(box['y1'] + box['y2'])))

class TrackFromBounds(Op):

    def __init__(self, decoder, window, trajectory_key='trajectory', name=None):
        super().__init__(name)
        assert isinstance(decoder, AbstractVideoDecoder)
        self.decoder = decoder
        self.window = window
        self.trajectory_key = trajectory_key

    def call(self, instream):
        self.instream = instream

    def execute(self):
        i1 = self.instream.get()
        if i1 is None:
            print("Exiting track op")
            return False
        
        tracker = cv2.TrackerCSRT_create()
        ret_bounds = i1.bounds
        ret_payload = {'trajectory': [VideoFrameInterval(i1.bounds, root_decoder=self.decoder), ]}

        # init tracker. For tracking, we must get whole frames
        init_frame = self.decoder.get_frame(i1['t1'])
        H, W = init_frame.shape[:2]
        # tracking box in cv2 is the form (x, y, w, h)
        init_box = np.array([i1['x1']*W, i1['y1']*H, _width(i1)*W, _height(i1)*H]).astype(np.int32)
        tracker.init(init_frame, tuple(init_box))

        # iterate frames and update tracker, get tracked result
        for ts in range(int(i1['t1']+1), min(int(i1['t1']+self.window), int(self.decoder.frame_count))):
            next_frame = self.decoder.get_frame(ts)
            (success, next_box) = tracker.update(next_frame)
            if success:
                x, y, w, h = next_box # pixel coord
                x1, y1, x2, y2 = x, y, x+w, y+h
                x1, y1, x2, y2 = x1/W, y1/H, x2/W, y2/H # relative coord
                next_bounds = Bounds3D(ts, ts, x1, x2, y1, y2)
                ret_bounds = ret_bounds.span(next_bounds)
                ret_payload['trajectory'].append(
                    VideoFrameInterval(next_bounds, root_decoder=self.decoder)
                )
            else:
                break
        
        self.publish(Interval(ret_bounds, ret_payload))
        return True


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fps = 15
    detect_every = 8
    
    all_frames = LocalVideoToFrames(INPUT_NAME)()
    sampled_frames = Slice(step=detect_every)(all_frames)
    detections = Detection('cloudlet031.elijah.cs.cmu.edu', 5000)(sampled_frames)
    crop_persons = DetectionFilterFlatten(['person'], 0.5)(detections)

    track_person_trajectories = TrackFromBounds(LRULocalVideoDecoder(INPUT_NAME), detect_every+1)(crop_persons)

    def trajectory_merge_predicate(i1, i2):
        return meets_before(1)(i1, i2) \
            and iou_at_least(0.5)(i1.payload['trajectory'][-1], i2.payload['trajectory'][0])

    def trajectory_payload_merge_op(p1, p2):
        print(f"Merging two trajectories of lengths {len(p1['trajectory'])} and {len(p2['trajectory'])}")
        return {'trajectory': p1['trajectory'] + p2['trajectory']}

    coalesced_trajectories = CoalesceByLast(
        predicate=trajectory_merge_predicate,
        bounds_merge_op=Bounds3D.span,
        payload_merge_op=trajectory_payload_merge_op,
        epsilon=1.1*detect_every
    )(track_person_trajectories)

    long_coalesced_persons = Filter(
        pred_fn=lambda intrvl: intrvl.bounds.length() >= fps * 5
    )(coalesced_trajectories)

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
                f1 = cv2.line(f1, tuple(p1), tuple(p2), color, thickness)
                f1 = cv2.putText(f1, str(j), tuple(p1), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0,0,255), 1, cv2.LINE_AA)
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

    logger.info(
        "You should find cropped .mp4 videos that 'bounds' a moving person."
        " We use trackers from OpenCV."
        " The trajectory is visualized."
    )