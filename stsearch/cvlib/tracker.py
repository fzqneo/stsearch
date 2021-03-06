import collections
import functools
import time
import typing

import cv2
from logzero import logger
import numpy as np

from rekall.bounds import Bounds3D
from rekall.predicates import _height, _iou, _width

from stsearch import Graph, Interval, Op
from stsearch.cvlib.detection import DEFAULT_DETECTION_KEY
from stsearch.op import Flatten, Map
from stsearch.parallel import ParallelFlatten, ParallelFlatten, ParallelMap
from stsearch.third_party.sorttrack.sort import Sort as SORTTrack
from stsearch.videolib import AbstractVideoDecoder, VideoFrameInterval


def _cv2_track_from_box(decoder, window, step, trajectory_key, backward=False, bidirectional=False) -> typing.Callable[[Interval,], Interval]:
    assert step > 0
    assert not (backward and bidirectional)

    # Either forward or backward, the current t1 is included in the output trajectory

    def new_fn_forward(i1: Interval) -> Interval:
        tracker = cv2.TrackerCSRT_create() 
        ret_bounds = i1.bounds
        ret_payload = {trajectory_key: [VideoFrameInterval(i1.bounds, root_decoder=decoder), ]}

        # buffer all frames in window at once
        start_fid = min(i1['t1'], decoder.frame_count - 1)  # inclusive
        end_fid = min(i1['t1'] + window, decoder.frame_count)  # exclusive
        frames_to_track = decoder.get_frame_interval(start_fid, end_fid, step)

        # init tracker. For tracking, we must get whole frames
        H, W = frames_to_track[0].shape[:2]
        # tracking box in cv2 is the form (x, y, w, h)
        init_box = np.array([i1['x1']*W, i1['y1']*H, _width(i1)*W, _height(i1)*H]).astype(np.int32)
        tracker.init(frames_to_track[0], tuple(init_box))

        # iterate remaining frames and update tracker, get tracked result
        for ts, next_frame in zip(range(start_fid+step, end_fid, step), frames_to_track[1:]):
            (success, next_box) = tracker.update(next_frame)

            if success:
                x, y, w, h = next_box # pixel coord
                x1, y1, x2, y2 = x, y, x+w, y+h
                x1, y1, x2, y2 = x1/W, y1/H, x2/W, y2/H # relative coord
                next_bounds = Bounds3D(ts, ts+1, x1, x2, y1, y2)
                ret_bounds = ret_bounds.span(next_bounds)
                ret_payload[trajectory_key].append(
                    VideoFrameInterval(next_bounds, root_decoder=decoder)
                )
            else:
                break
        
        return Interval(ret_bounds, ret_payload)

    def new_fn_backward(i1: Interval) -> Interval:
        tracker = cv2.TrackerCSRT_create()
        ret_bounds = i1.bounds
        ret_payload = {trajectory_key: [VideoFrameInterval(i1.bounds, root_decoder=decoder), ]}

        # buffer all frames in window at once
        ts_range = list(range(i1['t1'], max(-1, i1['t1']-window), -step))    # reverse order
        start_fid = min(ts_range)  # inclusive
        end_fid = max(ts_range) + 1  # exclusive
        frames_to_track = decoder.get_frame_interval(start_fid, end_fid, step)[::-1]    # reverse tracking order

        # init tracker. For tracking, we must get whole frames
        H, W = frames_to_track[0].shape[:2]
        # tracking box in cv2 is the form (x, y, w, h)
        init_box = np.array([i1['x1']*W, i1['y1']*H, _width(i1)*W, _height(i1)*H]).astype(np.int32)
        tracker.init(frames_to_track[0], tuple(init_box))

        # iterate remaining frames and update tracker, get tracked result
        for ts, next_frame in zip(ts_range[1:], frames_to_track[1:]):
            # tracking backward
            (success, next_box) = tracker.update(next_frame)

            if success:
                x, y, w, h = next_box # pixel coord
                x1, y1, x2, y2 = x, y, x+w, y+h
                x1, y1, x2, y2 = x1/W, y1/H, x2/W, y2/H # relative coord
                next_bounds = Bounds3D(ts, ts+1, x1, x2, y1, y2)
                ret_bounds = ret_bounds.span(next_bounds)
                ret_payload[trajectory_key].insert(
                    0,
                    VideoFrameInterval(next_bounds, root_decoder=decoder)
                )
            else:
                break
        
        return Interval(ret_bounds, ret_payload)

    def new_fn_bidirectional(i1: Interval) -> Interval:
        i_backward = new_fn_backward(i1)
        i_forward = new_fn_forward(i1)
        ret_bounds = i_backward.bounds.span(i_forward.bounds)
        ret_payload = {
            trajectory_key: i_backward.payload[trajectory_key][:-1] + i_forward.payload[trajectory_key]
        }
        return Interval(ret_bounds, ret_payload)
    
    if bidirectional:
        logger.debug("tracking bidirectional")
        return new_fn_bidirectional
    elif backward:
        logger.debug("tracking backward")
        return new_fn_backward
    else:
        logger.debug("tracking forward")
        return new_fn_forward

class TrackFromBox(Graph):

    def __init__(self, decoder, window, step=1, trajectory_key='trajectory', backward=False, bidirectional=False, name=None, parallel_workers=1):
        super().__init__()
        assert isinstance(decoder, AbstractVideoDecoder)
        self.decoder = decoder
        self.window = window
        self.step = step
        self.trajectory_key = trajectory_key
        self.name = name
        self.parallel_workers = parallel_workers
        self.backward = backward
        self.bidirectional = bidirectional

    def call(self, instream):
        if self.parallel_workers == 1:
            return Map(
                map_fn=_cv2_track_from_box(self.decoder, self.window, self.step, self.trajectory_key, self.backward, self.bidirectional),
                name=f"{self.__class__.__name__}:{self.name}"
            )(instream)
        else:
            return ParallelMap(
                map_fn=_cv2_track_from_box(self.decoder, self.window, self.step, self.trajectory_key, self.backward, self.bidirectional),
                name=f"{self.__class__.__name__}:{self.name}",
                max_workers=self.parallel_workers
            )(instream)


def get_boxes_from_detection(
    classes: typing.Collection[str] , 
    min_score, 
    detection_key=DEFAULT_DETECTION_KEY) -> typing.Callable[[Interval], np.ndarray]:
    """Returns an n_object x 5 array of [x_min, y_min, x_max, y_max, score].
    This format is used by multi-object trackers.
    Assume input is generated by stsearch.cvlib.detection.Detection.

    Args:
        classes ([type]): [description]
        min_score ([type]): [description]
    """

    def new_fn(intrvl: Interval) -> np.ndarray:
        ret = []
        detections = intrvl.payload[detection_key]
        for box, score, class_name in zip(detections['detection_boxes'], detections['detection_scores'], detections['detection_names']):
            if score < min_score:
                break
            if class_name in classes:
                # create new patch
                top, left, bottom, right = box  # TF return between 0~1
                ret.append([left, top, right, bottom, score])

        ret = np.array(ret, dtype=np.float) if ret else np.empty((0,5))
        return ret

    return new_fn


class SORTTrackByDetection(Op):

    # Hopefully relative coords work as well

    def __init__(
        self, 
        get_boxes_fn: typing.Callable[[Interval], np.ndarray], 
        window: int,
        trajectory_key:str = 'trajectory',
        name = None,
        max_age=1, min_hits=3, iou_threshold=0.3):

        super().__init__(name)

        assert callable(get_boxes_fn)
        self.get_boxes_fn = get_boxes_fn
        self.window = window
        self.trajectory_key = trajectory_key
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold 

        self.trackings = collections.defaultdict(list) # id -> list of boxes [t1, x1, y1, x2, y2]
        self.result_buffer = [] # Intervals with trajectory
        self.cur_t1 = float('-inf')  # t1 of the starting point of the current tracking
        self.tracker = None
        self.done = False

    def call(self, instream):
        self.instream = instream

    def execute(self):
        
        while not self.done and len(self.result_buffer) == 0:
            i1 = self.instream.get()
            if i1 is None or i1['t1'] > self.cur_t1 + self.window:
                # release pending trackings
                for _, v in self.trackings.items():
                    # v is a list of (t1, x1, y1, x2, y2)
                    new_trajectory = []
                    for t1, x1, y1, x2, y2 in v:
                        new_trajectory.append(
                            Interval(Bounds3D(t1, t1+1, x1, x2, y1, y2))
                        )

                    new_bounds = functools.reduce(Bounds3D.span, [i.bounds for i in new_trajectory])
                    new_payload = {self.trajectory_key: new_trajectory}
                    self.result_buffer.append(Interval(new_bounds, new_payload))

                self.result_buffer.sort(key=lambda i: i['t1'])
                self.trackings.clear()

                if i1 is None:
                    self.done = True
                    break
                else:
                    # reset tracker and cur_t1
                    self.tracker = SORTTrack(self.max_age, self.min_hits, self.iou_threshold)
                    self.cur_t1 = i1['t1']

            dets = self.tracker.update(self.get_boxes_fn(i1))
            assert dets.shape[1] == 5, str(dets)
            for x1, y1, x2, y2, oid in dets:
                self.trackings[oid].append([i1['t1'], x1, y1, x2, y2])            
        

        if len(self.result_buffer) > 0:
            self.publish(self.result_buffer.pop(0))
            return True
        else:
            return False


from .optical_flow import get_good_features_to_track, estimate_feature_translation, estimate_box_translation

_DEBUG_VIS_OPTICAL = False  # requires a vis/ folder in working directory
class TrackOpticalFlowFromBoxes(Graph):

    # Ad https://github.com/jguoaj/multi-object-tracking

    def __init__(
        self, 
        get_boxes_fn: typing.Callable[[Interval], np.ndarray], 
        decoder: AbstractVideoDecoder, 
        window, 
        step=1, 
        trajectory_key='trajectory',
        name = None,
        parallel=1,
        ):

        super().__init__()

        assert callable(get_boxes_fn)
        self.get_boxes_fn = get_boxes_fn
        self.decoder = decoder
        self.window = window
        self.step = step
        self.trajectory_key = trajectory_key
        self.name = name or f"{self.__class__.__name__}"
        self.parallel = parallel

    def call(self, instream):
        decoder = self.decoder
        window = self.window
        step = self.step

        def flatten_fn(i1: Interval) -> typing.List[Interval]:            
            # buffer all frames in window at once
            start_fid = min(i1['t1'], decoder.frame_count - 1)  # inclusive
            end_fid = min(i1['t1'] + window, decoder.frame_count)  # exclusive
            origin_frames_to_track = decoder.get_frame_interval(start_fid, end_fid, step)
            frames_to_track = [ cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in origin_frames_to_track ]  # optical flow only needs gray images

            old_frame = frames_to_track[0]
            H, W = old_frame.shape[:2]

            initial_boxes_and_score = self.get_boxes_fn(i1)
            if initial_boxes_and_score.shape[0] == 0:
                print("No box to start.")
                return []

            keep_track = [ [ (i1['t1'], xmin, xmax, ymin, ymax),] for xmin, ymin, xmax, ymax, _ in initial_boxes_and_score ]
            done_track = []

            # box format: (xmin, ymin, xmax, ymax)
            old_bboxs = np.multiply(initial_boxes_and_score[:, :4], [W, H, W, H]).astype(int)    # convert relative to pixel coord
            assert old_bboxs.shape[1] == 4

            old_features = get_good_features_to_track(old_frame, old_bboxs)
            assert len(old_features) == old_bboxs.shape[0]

            vis_img = cv2.cvtColor(origin_frames_to_track[0].copy(), cv2.COLOR_RGB2BGR)

            if _DEBUG_VIS_OPTICAL:
                for x,y in np.vstack(old_features):
                    x, y = int(x), int(y)
                    vis_img = cv2.circle(vis_img,(x,y),3, (255,0,0),-1)
                for xmin, ymin, xmax, ymax in old_bboxs:
                    vis_img = cv2.rectangle(vis_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
                cv2.imwrite(f"vis/{start_fid}-init.jpg", vis_img)

            for ts, frame, color_frame in \
                zip(range(start_fid + step, end_fid, step), frames_to_track[1:], origin_frames_to_track[1:] ):
        
                if len(keep_track) == 0:
                    break

                try:
                    good_old_features, good_new_features = estimate_feature_translation(old_frame, frame, old_features)
                except:
                    logger.error(old_features)
                    raise

                new_bboxs, status = estimate_box_translation(good_old_features, good_new_features, old_bboxs)
                
                if np.count_nonzero(status) == 0:
                    break

                temp_keep_track = []

                for track_id, (cur_track, box, s) in enumerate(zip(keep_track, new_bboxs, status)):
                    if s:
                        xmin, ymin, xmax, ymax = box
                        temp_keep_track.append(cur_track)
                        cur_track.append( (ts, xmin/W, xmax/W, ymin/H, ymax/H ))
                    else:
                        logger.info(f"Track {track_id} terminates at frame {ts}")
                        done_track.append(cur_track)

                new_bboxs = np.vstack(new_bboxs[status==1]).round().astype(int)
                new_bboxs[:, [0,2]] = np.clip(new_bboxs[:, [0,2]], 0, W)
                new_bboxs[:, [1,3]] = np.clip(new_bboxs[:, [1,3]], 0, H)
                new_features = [ good_new_features[i] for i in np.nonzero(status)[0] ]
                assert len(new_features) == new_bboxs.shape[0] == len(temp_keep_track), f"{len(new_features)}, {new_bboxs.shape}, {len(temp_keep_track)}"
           
                if _DEBUG_VIS_OPTICAL:
                    vis_img = cv2.cvtColor(color_frame.copy(), cv2.COLOR_RGB2BGR)
                    for (a,b), (c,d) in zip(np.vstack(good_old_features), np.vstack(good_new_features)):
                        a, b, c, d = int(a), int(b), int(c), int(d)
                        # vis_img = cv2.circle(vis_img,(a,b),3, (255,0,0),-1)
                        vis_img = cv2.circle(vis_img,(c,d),3, (0,255,0),-1)
                        vis_img = cv2.line(vis_img, (a,b), (c,d), (0,0,255), 2)
                    for xmin, ymin, xmax, ymax in old_bboxs:
                        vis_img = cv2.rectangle(vis_img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
                    for xmin, ymin, xmax, ymax in new_bboxs:
                        vis_img = cv2.rectangle(vis_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
                    cv2.imwrite(f"vis/{ts}.jpg", vis_img)

                # update old stuff
                keep_track = temp_keep_track
                old_bboxs = new_bboxs
                old_features = new_features
                old_frame = frame

            # Done. Convert keep_track into output
            rv = []
            for track in (done_track + keep_track):
                new_trajectory = [
                    Interval(
                        Bounds3D(ts, ts+1, xmin, xmax, ymin, ymax)
                    )
                    for ts, xmin, xmax, ymin, ymax in track
                ]
                new_bounds = functools.reduce(Bounds3D.span, [i.bounds for i in new_trajectory])
                new_payload = {self.trajectory_key: new_trajectory}
                rv.append(Interval(new_bounds, new_payload))

            return rv # end of flatten_fn
        
        if self.parallel == 1:
            return Flatten(flatten_fn, self.name)(instream)
        else:
            return ParallelFlatten(flatten_fn, self.name, max_workers=self.parallel)(instream)
