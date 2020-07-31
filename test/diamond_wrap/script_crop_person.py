from stsearch.cvlib import Detection, DetectionFilter, DetectionFilterFlatten
from stsearch.interval import *
from stsearch.op import *
from stsearch.utils import run_to_finish
from stsearch.videolib import *

def query(path):
    rv = []

    all_frames = LocalVideoToFrames(path)()
    sampled_frames = Slice(step=30, end=300)(all_frames)
    detections = Detection('cloudlet031.elijah.cs.cmu.edu', 5000)(sampled_frames)
    person_detections = DetectionFilterFlatten(['person'], 0.3)(detections)

    for _, intrvl in enumerate(run_to_finish(person_detections)):
        rv.append((intrvl.bounds, intrvl.jpeg))

    return rv