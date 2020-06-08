from stsearch.interval import *
from stsearch.video import *
from stsearch.op import *

if __name__ == "__main__":
    
    all_frames = LocalVideoToFrames("VIRAT_S_000200_02_000479_000635.mp4")()
    sampled_frames = Slice(1, None, 30)(all_frames)
    cropped_frames = Crop(.5, .8, .5, .8)(sampled_frames)
    
    assert isinstance(cropped_frames, IntervalStream)
    for k in range(30):
        ii = cropped_frames.get()
        ii.savefile(f"output/test_video_crop_{k}.jpg")