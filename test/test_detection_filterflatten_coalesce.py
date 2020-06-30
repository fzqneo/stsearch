from rekall.bounds import Bounds3D
from rekall.predicates import iou_at_least

from stsearch.cvlib import Detection, DetectionFilter, DetectionFilterFlatten
from stsearch.interval import *
from stsearch.op import *
from stsearch.utils import run_to_finish
from stsearch.videolib import *

if __name__ == "__main__":
    path = "FifthCraig1-2019-02-01-10-05-05.mp4"
    fps = 15
    sample_every = 10
    
    all_frames = LocalVideoToFrames(path)()
    sampled_frames = Slice(step=sample_every)(all_frames)
    detections = Detection('cloudlet015.elijah.cs.cmu.edu', 5000)(sampled_frames)
    person_detections = DetectionFilterFlatten(['person'], 0.5)(detections)
    coalesced_persons = Coalesce(
        bounds_merge_op=Bounds3D.span,
        predicate=iou_at_least(0.5),
        epsilon=sample_every)(person_detections)
    long_coalesced_persons = Filter(
        pred_fn=lambda intrvl: intrvl.bounds.length() > fps * 2 # 2 seconds
    )(coalesced_persons)
    fg = LocalVideoCropInterval(path)(long_coalesced_persons)

    output = fg
    output_sub = output.subscribe()
    output.start_thread_recursive()

    for k, intrvl in enumerate(output_sub):
        assert isinstance(intrvl, FrameGroupInterval)
        print(intrvl.bounds)
        intrvl.savevideo(f"output_detection_filterflatten_coalesce/{k:06d}.mp4", fps=fps)
