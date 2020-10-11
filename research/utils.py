import cv2
import numpy as np
from operator import itemgetter

from rekall.predicates import _area, _height, _width

from stsearch.op import Graph, Map, Op
from stsearch.parallel import ParallelMap
from stsearch.stdlib import centroid
from stsearch.videolib import FrameGroupInterval

class VisualizeTrajectoryOnFrameGroup(Graph):

    def __init__(
        self, 
        trajectory_key='trajectory', parallel=1, 
        draw_line=True, draw_label=False, draw_box=True,
        name=None):
        super().__init__()
        self.trajectory_key = trajectory_key
        self.parallel = parallel
        self.name = name
        self.draw_line = draw_line
        self.draw_label = draw_label
        self.draw_box = draw_box

    def call(self, instream):
        tkey = self.trajectory_key

        def visualize_map_fn(fg: FrameGroupInterval):
            assert isinstance(fg, FrameGroupInterval)
            assert tkey in fg.payload, f"{tkey} not found in {str(fg)}"
            assert fg['t2'] - fg['t1'] == len(fg.frames)

            # 1. Draw all visualization on a frame with black background
            global_vis_frame = np.zeros_like(fg.frames[0])

            trajectory = fg.payload[tkey]
            Xmins = list(map(itemgetter('x1'), trajectory))
            Xmaxs = list(map(itemgetter('x2'), trajectory))
            Ymins = list(map(itemgetter('y1'), trajectory))
            Ymaxs = list(map(itemgetter('y2'), trajectory))
            Xmins, Xmaxs, Ymins, Ymaxs = list(map(np.array, [Xmins, Xmaxs, Ymins, Ymaxs]))

            # this is tricky: adjust from relative coord in original frame to pixel coord in the crop
            H, W = fg.frames[0].shape[:2]
            Xmins = W * (Xmins - fg['x1']) / _width(fg)
            Xmaxs = W * (Xmaxs - fg['x1']) / _width(fg)
            Ymins = H * (Ymins - fg['y1']) / _height(fg)
            Ymaxs = H * (Ymaxs - fg['y1']) / _height(fg)

            Xmins = Xmins.astype(np.int)
            Xmaxs = Xmaxs.astype(np.int)
            Ymins = Ymins.astype(np.int)
            Ymaxs = Ymaxs.astype(np.int)

            centroids = np.stack([0.5 * (Xmins + Xmaxs), 0.5 * (Ymins + Ymaxs)], axis=1)
            centroids = centroids.astype(np.int)
            assert centroids.shape == (len(trajectory), 2)

            # somehow polylines doesn't work
            # f1 = cv2.polylines(f1, pts, is_closed, color, thickness)
            color = (0, 255, 0)
            thickness = 2
            is_closed = False

            for j, (p1, p2) in enumerate(zip(centroids[:-1], centroids[1:])):
                if self.draw_line:
                    global_vis_frame = cv2.line(global_vis_frame, tuple(p1), tuple(p2), color, thickness)
                if self.draw_label:
                    global_vis_frame = cv2.putText(global_vis_frame, str(j), tuple(p1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

            if self.draw_box:
                boxs_to_draw = list(zip([i['t1'] for i in trajectory], Xmins, Ymins, Xmaxs, Ymaxs))

            # 2. Overlay it on all frames
            new_frames = []
            vis_ts_box = None
            for j, frame in enumerate(fg.frames):
                frame = frame.copy()

                ts = fg['t1'] + j
                frame = cv2.putText(frame, f"F {ts}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

                assert global_vis_frame.shape == frame.shape, f"{j} {global_vis_frame.shape} {frame.shape}"
                frame[global_vis_frame > 0] = global_vis_frame[global_vis_frame > 0] 

                if self.draw_box:
                    for b in boxs_to_draw:
                        if ts == b[0]:
                            vis_ts_box = b

                    if vis_ts_box:
                        _, left, top, right, bottom = vis_ts_box
                        frame = cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 3)

                new_frames.append(frame)


            fg.frames = new_frames
            return fg

        return ParallelMap(
            map_fn=visualize_map_fn,
            name=self.name or f"{self.__class__.__name__}:{self.name}",
            max_workers=self.parallel)(instream)