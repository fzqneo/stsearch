import os
from pathlib import Path

from logzero import logger

from rekall.bounds import Bounds3D
from rekall.predicates import iou_at_least

from stsearch.cvlib import Detection, DetectionFilter, DetectionFilterFlatten
from stsearch.interval import *
from stsearch.op import *
from stsearch.utils import run_to_finish
from stsearch.videolib import *

INPUT_NAME = "example.mp4"
OUTPUT_DIR = Path(__file__).stem + "_output"

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fps = 15
    sample_every = 3
    
    all_frames = LocalVideoToFrames(INPUT_NAME)()
    sampled_frames = Slice(step=sample_every)(all_frames)
    detections = Detection('cloudlet031.elijah.cs.cmu.edu', 5000)(sampled_frames)
    crop_persons = DetectionFilterFlatten(['person'], 0.3)(detections)

    # Try `Coalesce()`. Will have different results
    coalesced_persons = CoalesceByLast(
        bounds_merge_op=Bounds3D.span,
        predicate=iou_at_least(0.1),
        epsilon=sample_every*3)(crop_persons)

    long_coalesced_persons = Filter(
        pred_fn=lambda intrvl: intrvl.bounds.length() > fps * 3 # 3 seconds
    )(coalesced_persons)
    fg = LocalVideoCropFrameGroup(INPUT_NAME)(long_coalesced_persons)

    output = fg
    output_sub = output.subscribe()
    output.start_thread_recursive()

    for k, intrvl in enumerate(output_sub):
        assert isinstance(intrvl, FrameGroupInterval)
        out_name = f"{OUTPUT_DIR}/{k}-{intrvl['t1']}-{intrvl['t2']}-{intrvl['x1']:.2f}-{intrvl['y1']:.2f}.mp4"
        intrvl.savevideo(out_name, fps=fps)
        logger.debug(f"saved {out_name}")

    logger.info('You should find cropped .mp4 videos that "bounds" a moving person. We use Coalesce/CoalesceByLast with iou predicate to do a form of tracking.')