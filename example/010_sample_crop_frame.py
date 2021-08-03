import os
from pathlib import Path

from logzero import logger

import stsearch
from stsearch.interval import *
from stsearch.op import *
from stsearch.videolib import *

INPUT_NAME = "example.mp4"
OUTPUT_DIR = Path(__file__).stem + "_output"

if __name__ == "__main__":
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    stsearch.exc = None

    all_frames = VideoToFrames(LocalVideoDecoder(INPUT_NAME))()
    sampled_frames = Slice(start=0, step=30, end=900)(all_frames)
    cropped_frames = Crop(x1=.25, x2=.5, y1=.25, y2=.75)(sampled_frames)
    
    results = cropped_frames.subscribe()
    cropped_frames.start_thread_recursive()

    for k, ii in enumerate(results):
        assert isinstance(ii, ImageInterval)
        output_name = f"{OUTPUT_DIR}/{k}.jpg"
        ii.savefile(output_name)
        logger.debug(f"saved {output_name}")

    if stsearch.exc:
        raise stsearch.exc

    logger.info("You should find 30 .jpg files sampled from 900 timesteps, each cropped at center.")