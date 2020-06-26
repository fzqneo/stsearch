from stsearch.interval import *
from stsearch.op import *
from stsearch.videolib import *

if __name__ == "__main__":
    
    all_frames = LocalVideoToFrames("VIRAT_S_000200_02_000479_000635.mp4")()
    sampled_frames = Slice(step=30)(all_frames)
    cropped_frames = Crop(x1=.25, x2=.5, y1=.25, y2=.75)(sampled_frames)
    
    results = cropped_frames.subscribe()
    cropped_frames.start_thread_recursive()

    for k, ii in enumerate(iter(results)):
        print(ii['t1'], ii['t2'])
        ii.savefile(f"output/test_video_crop_{k}.jpg")
