import os
from pathlib import Path

from logzero import logger

from stsearch.cvlib import Detection, DetectionFilter, DetectionFilterFlatten
from stsearch.interval import *
from stsearch.op import *
from stsearch.utils import run_to_finish
from stsearch.videolib import *

INPUT_NAME = "example.mp4"
OUTPUT_DIR = Path(__file__).stem + "_output"

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_frames = VideoToFrames(LocalVideoDecoder(INPUT_NAME))()
    sampled_frames = Slice(step=30, end=450)(all_frames)
    detections = Detection('cloudlet031.elijah.cs.cmu.edu', 5000)(sampled_frames)
    # `DetectionFilterFlatten` separates the patches while `DetectionFilter` just filters whole frames.
    person_crops = DetectionFilterFlatten(['person'], 0.5)(detections)
    large_person_crops = Filter(
        pred_fn=lambda ii: ii.bounds.height() > 0.05
    )(person_crops)
    output = large_person_crops

    results = output.subscribe()
    output.start_thread_recursive()

    for k, intrvl in enumerate(results):
        out_name = f"{OUTPUT_DIR}/{k}-{intrvl['t1']}-{intrvl['x1']:.2f}-{intrvl['y1']:.2f}.jpg"
        intrvl.savefile(out_name)
        logger.debug(f"saved {out_name}")

    logger.info("You should find .jpg files of large cropped patches of persons")
