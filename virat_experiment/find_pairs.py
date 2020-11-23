from rekall.bounds import Bounds3D
from rekall.predicates import _area, _height, _width, meets_before, iou_at_least, overlaps_before

from stsearch.cvlib import Detection, DetectionFilterFlatten, TrackFromBox
from stsearch.interval import *
from stsearch.op import *
from stsearch.stdlib import centroid, same_time
from stsearch.utils import run_to_finish
from stsearch.videolib import *

DETECTION_SERVERS = ["172.17.0.1:5000", "172.17.0.1:5001"]

# VideoCapture seems to be causing memory leak https://github.com/opencv/opencv/issues/13255

def is_pair(corrcoef=.8, trajectory_key='trajectory'):

    def new_pred(i1: Interval, i2: Interval) -> bool:
        assert trajectory_key in i1.payload
        assert trajectory_key in i2.payload

        if not id(i1) < id(i2) \
            or not same_time(30*10)(i1, i2) \
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


import threading

def query(path, session):
    cv2.setNumThreads(8)
    # session.log('error', f"starting on path {path}. running threads: {len(threading.enumerate())}")

    query_result = {}

    decoder = LocalVideoDecoder(path)
    frame_count, fps = decoder.frame_count, int(np.round(decoder.fps))
    query_result['metadata'] = {
        'fps': fps,
        'frame_count': frame_count,
        'raw_w': decoder.raw_width,
        'raw_h': decoder.raw_height,
    }
    query_result['results'] = list()
    del decoder

    detect_step = int(fps)

    all_frames = VideoToFrames(LocalVideoDecoder(path))()
    sampled_frames = Slice(step=detect_step)(all_frames)
    detections = Detection(server_list=DETECTION_SERVERS, parallel=2)(sampled_frames)
    crop_persons = DetectionFilterFlatten(['person'], 0.5)(detections)

    track_trajectories = TrackFromBox(
        LRULocalVideoDecoder(path, cache_size=900), 
        detect_step,
        step=2,
        name="track_person",
    )(crop_persons)

    def trajectory_merge_predicate(i1, i2):
        return meets_before(detect_step*2)(i1, i2) \
            and iou_at_least(0.1)(i1.payload['trajectory'][-1], i2.payload['trajectory'][0])

    def trajectory_payload_merge_op(p1, p2):
        # logger.debug(f"Merging two trajectories of lengths {len(p1['trajectory'])} and {len(p2['trajectory'])}")
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

    raw_fg = VideoCropFrameGroup(LRULocalVideoDecoder(path), copy_payload=True)(pairs)
    
    output = raw_fg
    output_sub = output.subscribe()
    output.start_thread_recursive()

    for _, intrvl in enumerate(output_sub):
        assert isinstance(intrvl, FrameGroupInterval)
        # RAM still goes up without mp4
        query_result['results'].append((intrvl.bounds.copy(), intrvl.get_mp4()))
        del intrvl
        # query_result['results'].append((intrvl.bounds, b''))
        # session.log('error', "Find a pair!")
        
    return query_result


if __name__ == "__main__":
    from pathlib import Path
    import pickle
    import time

    from stsearch.diamond_wrap.result_pb2 import STSearchResult
    from utils import start_stsearch_by_script, OUTPUT_ATTR

    tic = time.time()

    results = start_stsearch_by_script(open(__file__, 'rb').read())

    for i, res in enumerate(results):
        print(f"=> Result {i}. Time {(time.time()-tic)/60} min.")

        object_id = res['_ObjectID'].decode()
        clip_id = Path(object_id).stem

        filter_result: STSearchResult = STSearchResult()
        filter_result.ParseFromString(res[OUTPUT_ATTR])
        query_result = pickle.loads(filter_result.query_result)
        print(f"{clip_id}, {filter_result.stats}, {query_result['metadata']}. #={len(query_result['results'])}")
        for b, mp4 in query_result['results']:
            open(f"pairs_{clip_id}_{b['t1']}_{b['t2']}.mp4", 'wb').write(mp4)




