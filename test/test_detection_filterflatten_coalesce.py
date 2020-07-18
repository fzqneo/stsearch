import math

from rekall.bounds import Bounds3D
from rekall.predicates import iou_at_least

from stsearch.cvlib import Detection, DetectionFilter, DetectionFilterFlatten
from stsearch.interval import *
from stsearch.op import *
from stsearch.utils import run_to_finish
from stsearch.videolib import *


# def center(i):
#     return (.5*(i['x1'] + i['x2']), .5*(i['y1'] + i['y2']))


# def center_l2(i1, i2):
#     c1, c2 = center(i1), center(i2)
#     return math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)

# def coalesce_tail(tail_predicate, *args, **kwargs):

#     def pred_fn(cur_i, new_i):
#         last = cur_i.payload.get('last', cur_i)
#         return tail_predicate(last, new_i)

#     def interval_merge_op(cur_i, new_i):
#         merged_i = Interval(
#             Bounds3D.span(cur_i.bounds, new_i.bounds),
#             cur_i.payload
#         )

#         merged_i.payload['last'] = new_i
#         return merged_i
    
#     return Coalesce(*args, **kwargs, interval_merge_op=interval_merge_op, predicate=pred_fn)


if __name__ == "__main__":
    path = "FifthCraig_sample_1.mp4"
    fps = 15
    sample_every = 3
    
    all_frames = LocalVideoToFrames(path)()
    sampled_frames = Slice(step=sample_every)(all_frames)
    detections = Detection('localhost', 5000)(sampled_frames)
    person_detections = DetectionFilterFlatten(['person'], 0.3)(detections)

    coalesced_persons = CoalesceByLast(
        bounds_merge_op=Bounds3D.span,
        predicate=iou_at_least(0.1),
        epsilon=sample_every*3)(person_detections)

    long_coalesced_persons = Filter(
        pred_fn=lambda intrvl: intrvl.bounds.length() > fps * 3 # 3 seconds
    )(coalesced_persons)
    fg = LocalVideoCropInterval(path)(long_coalesced_persons)

    output = fg
    output_sub = output.subscribe()
    output.start_thread_recursive()

    for k, intrvl in enumerate(output_sub):
        assert isinstance(intrvl, FrameGroupInterval)
        print(intrvl.bounds)
        intrvl.savevideo(f"output_detection_filterflatten_coalesce/{k:06d}-{intrvl['t1']}-{intrvl['t2']}.mp4", fps=fps)
