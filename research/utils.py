import cv2
import numpy as np

from rekall.predicates import _area, _height, _width

from stsearch.op import Graph, Map, Op
from stsearch.parallel import ParallelMap
from stsearch.stdlib import centroid
from stsearch.videolib import FrameGroupInterval

class VisualizeTrajectoryOnFrameGroup(Graph):

    def __init__(self, trajectory_key='trajectory', parallel=1, name=None):
        super().__init__()
        self.trajectory_key = trajectory_key
        self.parallel = parallel
        self.name = name

    def call(self, instream):
        tkey = self.trajectory_key

        def visualize_map_fn(intrvl):
            assert isinstance(intrvl, FrameGroupInterval)
            assert tkey in intrvl.payload, f"{tkey} not found in {str(intrvl)}"
            pts = np.array(list(map(centroid, intrvl.payload[tkey])))
            assert pts.shape[1] == 2
            # this is tricky: adjust from relative coord in original frame to pixel coord in the crop
            pts =   (pts - [intrvl['x1'], intrvl['y1']]) / [_width(intrvl), _height(intrvl)] *  [intrvl.frames[0].shape[1], intrvl.frames[0].shape[0]]
            pts = pts.astype(np.int32)

            color = (0, 255, 0)
            thickness = 2
            is_closed = False

            new_frames = []

            for fid, frame in enumerate(intrvl.frames):
                frame = frame.copy()
                f1 = cv2.putText(frame, f"visualize-{fid}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                # somehow polylines doesn't work
                # f1 = cv2.polylines(f1, pts, is_closed, color, thickness)
                for j, (p1, p2) in enumerate(zip(pts[:-1], pts[1:])):
                    f1 = cv2.line(f1, tuple(p1), tuple(p2), color, thickness)
                    f1 = cv2.putText(f1, str(j), tuple(p1), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0,0,255), 1, cv2.LINE_AA)
                new_frames.append(f1)

            intrvl.frames = new_frames
            return intrvl

        return ParallelMap(
            map_fn=visualize_map_fn,
            name=self.name or f"{self.__class__.__name__}:{self.name}",
            max_workers=self.parallel)(instream)