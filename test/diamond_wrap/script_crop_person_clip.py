from stsearch.cvlib import Detection, DetectionFilter, DetectionFilterFlatten
from stsearch.interval import *
from stsearch.op import *
from stsearch.utils import run_to_finish
from stsearch.videolib import *

def query(path):
    rv = []

    fps = 30
    sample_every = 3
    
    all_frames = LocalVideoToFrames(path)()
    sampled_frames = Slice(step=sample_every, end=300)(all_frames)
    detections = Detection(server_list=['cloudlet031.elijah.cs.cmu.edu:5000', 'cloudlet031.elijah.cs.cmu.edu:5001'])(sampled_frames)
    person_detections = DetectionFilterFlatten(['person'], 0.3)(detections)

    coalesced_persons = CoalesceByLast(
        bounds_merge_op=Bounds3D.span,
        predicate=iou_at_least(0.1),
        epsilon=sample_every*3)(person_detections)

    long_coalesced_persons = Filter(
        pred_fn=lambda framegrps: framegrps.bounds.length() > fps * 3 # 3 seconds
    )(coalesced_persons)
    framegrps = LocalVideoCropInterval(path)(long_coalesced_persons)

    for _, fg in enumerate(run_to_finish(framegrps)):
        rv.append((fg.bounds, fg.get_mp4(), 'mp4'))

    return rv