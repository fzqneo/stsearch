import numpy as np
import operator

from rekall.bounds import Bounds3D
from rekall.predicates import overlaps

def centroid(box):
    return np.array([0.5*(box['x1'] + box['x2']), 0.5*(box['y1'] + box['y2'])])

def same_time(epsilon):
    return lambda i1, i2: abs(i1['t1']-i2['t1']) <= epsilon and abs(i1['t2']-i2['t2']) <= epsilon

def tiou(box1, box2) -> float:
    # IoU in t1, t2
    if not overlaps()(box1, box2):
        return 0

    U = max(box1['t2'], box2['t2']) - min(box1['t1'], box2['t1'])
    I = (box1['t2']-box1['t1']) + (box2['t2']-box2['t1']) - U
    assert U >= 0 and I >= 0
    return I/U

def tiou_at_least(thres):

    def new_pred(i1, i2) -> bool:
        return tiou(i1, i2) >= thres

    return new_pred

def average_space_span_time(list_of_box):
    ret_bounds = Bounds3D(
        t1=min([b['t1'] for b in list_of_box]),
        t2=max([b['t2'] for b in list_of_box]),
    )

    for key in ('x1', 'x2', 'y1', 'y2'):
        ret_bounds[key] = np.mean([b[key] for b in list_of_box])

    return ret_bounds

