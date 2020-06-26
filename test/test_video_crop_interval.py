from rekall.bounds import Bounds3D

from stsearch.interval import *
from stsearch.op import *
from stsearch.videolib import *
from stsearch.utils import run_to_finish

if __name__ == "__main__":

    intervals_to_crop = [
        Interval(Bounds3D(30, 330, 0, .5, 0, .5), {'msg': "0:01~0:11 upper left crop"}),
        Interval(Bounds3D(300, 1200, .25, .75, .4, .6), {'msg': "0:10~0:40 center crop"})
    ]

    crop_intervals = FromIterable(intervals_to_crop)()
    framegroups = LocalVideoCropInterval("FifthCraig1-2019-02-01-10-05-05.mp4")(crop_intervals)

    for j, fg in enumerate(run_to_finish(framegroups)):
        # print(fg)
        fg.savevideo(f"output_mp4/video_crop_interval_{j}.avi")
