import os
from pathlib import Path

from logzero import logger

from stsearch.cvlib import Detection, DetectionFilter
from stsearch.interval import *
from stsearch.op import *
from stsearch.utils import run_to_finish
from stsearch.videolib import *

INPUT_NAME = "example.mp4"
OUTPUT_DIR = Path(__file__).stem + "_output"

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    det_key = "03_just_some_unique_key"
    
    all_frames = LocalVideoToFrames(INPUT_NAME)()
    sampled_frames = Slice(step=30, end=1800)(all_frames)
    detections = Detection(
        'cloudlet031.elijah.cs.cmu.edu', 
        5000, 
        result_key=det_key)(sampled_frames)
    frames_with_target = DetectionFilter(['bus'], 0.5, result_key=det_key)(detections)

    results = frames_with_target.subscribe()
    frames_with_target.start_thread_recursive()

    logger.info("This may take a while because we run detection")
    for k, intrvl in enumerate(results):
        out_name = f"{OUTPUT_DIR}/{k}-{intrvl['t1']}.jpg"
        intrvl.savefile(out_name)
        logger.debug(f"saved {out_name}")

    logger.info("You should find .jpg files of frames with bus in it.")