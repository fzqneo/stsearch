from stsearch.interval import *
from stsearch.video import *
from stsearch.op import *

if __name__ == "__main__":
    
    all_frames = LocalVideoToFrames("VIRAT_S_000200_02_000479_000635.mp4")()
    sampled_frames = Slice(1, None, 300)(all_frames)
    cropped_frames = Crop(.25, .75, .25, .75)(sampled_frames)
    
    assert isinstance(cropped_frames, IntervalStream)

    for k, ii in enumerate(iter(cropped_frames)):
        print(ii['t1'])
        ii.savefile(f"output/test_video_crop_{k}.jpg")
