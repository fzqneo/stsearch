import cv2
import numpy as np

from rekall.bounds import Bounds3D
from rekall.predicates import _height, _iou, _width

from stsearch import Interval, Op
from stsearch.videolib import AbstractVideoDecoder, VideoFrameInterval

class TrackFromBox(Op):

    def __init__(self, decoder, window, step=1, trajectory_key='trajectory', name=None):
        super().__init__(name)
        assert isinstance(decoder, AbstractVideoDecoder)
        self.decoder = decoder
        self.window = window
        self.step = step
        self.trajectory_key = trajectory_key

    def call(self, instream):
        self.instream = instream

    def execute(self):
        i1 = self.instream.get()
        if i1 is None:
            return False
        
        tracker = cv2.TrackerCSRT_create() # best accuracy but slow
        # tracker = cv2.TrackerKCF_create()
        ret_bounds = i1.bounds
        ret_payload = {self.trajectory_key: [VideoFrameInterval(i1.bounds, root_decoder=self.decoder), ]}

        # init tracker. For tracking, we must get whole frames
        init_frame = self.decoder.get_frame(i1['t1'])
        H, W = init_frame.shape[:2]
        # tracking box in cv2 is the form (x, y, w, h)
        init_box = np.array([i1['x1']*W, i1['y1']*H, _width(i1)*W, _height(i1)*H]).astype(np.int32)
        tracker.init(init_frame, tuple(init_box))

        # iterate frames and update tracker, get tracked result
        for ts in range(int(i1['t1']+1), min(int(i1['t1']+self.window), int(self.decoder.frame_count)), int(self.step)):
            next_frame = self.decoder.get_frame(ts)
            (success, next_box) = tracker.update(next_frame)
            if success:
                x, y, w, h = next_box # pixel coord
                x1, y1, x2, y2 = x, y, x+w, y+h
                x1, y1, x2, y2 = x1/W, y1/H, x2/W, y2/H # relative coord
                next_bounds = Bounds3D(ts, ts, x1, x2, y1, y2)
                ret_bounds = ret_bounds.span(next_bounds)
                ret_payload[self.trajectory_key].append(
                    VideoFrameInterval(next_bounds, root_decoder=self.decoder)
                )
            else:
                break
        
        self.publish(Interval(ret_bounds, ret_payload))
        return True


class SORTTrackFromDetection(Op):

    pass