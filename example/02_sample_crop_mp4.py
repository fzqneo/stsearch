import os
from pathlib import Path

from logzero import logger
from rekall.bounds import Bounds3D

from stsearch.interval import *
from stsearch.op import *
from stsearch.utils import run_to_finish
from stsearch.videolib import *

INPUT_NAME = "example.mp4"
OUTPUT_DIR = Path(__file__).stem + "_output"

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    intervals_to_crop = [
        Interval(Bounds3D(t1=30, t2=330, x1=0, x2=.5, y1=0, y2=.5), {'msg': "ts 30-330 upper left crop"}),
        Interval(Bounds3D(300, 420, .25, .75, .4, .6), {'msg': "ts 300-420 center crop"})
    ]

    crop_intervals = FromIterable(intervals_to_crop)()
    framegroups = LocalVideoCropInterval(INPUT_NAME)(crop_intervals)

    for k, fg in enumerate(run_to_finish(framegroups)):
        assert isinstance(fg, FrameGroupInterval)
        out_name = f"{OUTPUT_DIR}/{k}-{fg['t1']}-{fg['t2']}.mp4"
        fg.savevideo(out_name)
        logger.debug(f"saved {out_name}")
    
    logger.info("You should find 2 .mp4 files, cropped -- spatially and temporally --  from the input.")
