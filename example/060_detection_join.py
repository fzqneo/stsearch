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

    # this is a fork in the compute graph
    crop_persons = DetectionFilterFlatten(['person'], 0.5)(detections)
    crop_cars = DetectionFilterFlatten(['car'], 0.5)(detections)

    crop_person_and_car = JoinWithTimeWindow(
        predicate=lambda i1, i2: i1['t1'] == i2['t1'] and iou_at_least(0.01)(i1, i2),
        merge_op=lambda i1, i2: ImageInterval(i1.bounds.span(i2), root=i1.root),
        window=1
    )(crop_persons, crop_cars)

    output = crop_person_and_car
    output_sub = output.subscribe()
    output.start_thread_recursive()

    logger.info("This may take a while")
    for k, intrvl in enumerate(output_sub):
        assert isinstance(intrvl, ImageInterval)
        out_name = f"{OUTPUT_DIR}/{k}-{intrvl['t1']}-{intrvl['t2']}-{intrvl['x1']:.2f}-{intrvl['y1']:.2f}.jpg"
        intrvl.savefile(out_name)
        logger.debug(f"saved {out_name}")

    logger.info('You should find cropped .jpg files containing both person and car.')