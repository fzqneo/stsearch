from stsearch.cvlib import Detection, DetectionFilter, DetectionFilterFlatten
from stsearch.interval import *
from stsearch.op import *
from stsearch.utils import run_to_finish
from stsearch.videolib import *

from stsearch.diamond_wrap.diamond_op import RGBDiamondSearch

from opendiamond.filter import Session
from opendiamond.client.search import Blob, FilterSpec


def query(path, session, *args, **kwargs):
    assert isinstance(session, Session)
    session.log('error', "Enter query fn")
    rv = []

    all_frames = LocalVideoToFrames(path)()
    sampled_frames = Slice(step=30, end=1800)(all_frames)
    detections = Detection('cloudlet031.elijah.cs.cmu.edu', 5000)(sampled_frames)
    person_detections = DetectionFilterFlatten(['bus'], 0.3)(detections)


    # SIFT filter
    fil_sift_extract_spec = FilterSpec(
        "sift-extract",
        code=Blob(open("fil_homography.py", 'rb').read()),
        arguments=['ExtractFilter', 'SIFT'],
        min_score=1.
    )

    fil_sift_match_spec = FilterSpec(
        "sift-match",
        code=Blob(open("fil_homography.py", 'rb').read()),
        arguments=['MatchFilter', 'SIFT', 0.7],
        dependencies=[fil_sift_extract_spec, ],
        blob_argument=Blob(open("brandenburg_hi_blob.zip", 'rb').read()),
        min_score=5
    )

    matched_patches = RGBDiamondSearch(session, [fil_sift_extract_spec, fil_sift_match_spec])(person_detections)

    for _, intrvl in enumerate(run_to_finish(matched_patches)):
        rv.append((intrvl.bounds, intrvl.jpeg, 'jpg'))

    return rv
