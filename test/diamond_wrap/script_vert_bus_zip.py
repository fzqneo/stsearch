from stsearch.cvlib import Detection, DetectionFilter, DetectionFilterFlatten
from stsearch.interval import *
from stsearch.op import *
from stsearch.utils import run_to_finish
from stsearch.videolib import *

from stsearch.diamond_wrap.diamond_op import RGBDiamondSearch

from opendiamond.filter import Session
from opendiamond.client.search import Blob, FilterSpec

FIL_ORIENTATION_CODE = open("fil_orientation.py", "rb").read()

def query(path, session, *args, **kwargs):
    assert isinstance(session, Session)
    session.log('error', "Enter query fn")
    rv = []

    all_frames = VideoToFrames(LocalVideoDecoder(path))()
    sampled_frames = Slice(step=30, end=1800)(all_frames)
    detections = Detection('cloudlet031.elijah.cs.cmu.edu', 5000)(sampled_frames)
    person_detections = DetectionFilterFlatten(['bus'], 0.3)(detections)

    # filter vertical 'vert' or 'horz'
    orient_spec = FilterSpec(
        "orient", Blob(FIL_ORIENTATION_CODE), arguments=['vert',], min_score=1.
    )

    oriented_patches = RGBDiamondSearch(session, [orient_spec, ])(person_detections)

    for _, intrvl in enumerate(run_to_finish(oriented_patches)):
        rv.append((intrvl.bounds, intrvl.jpeg, 'jpg'))

    return rv
