from stsearch.cvlib import Detection, DetectionFilter
from stsearch.interval import *
from stsearch.op import *
from stsearch.utils import run_to_finish
from stsearch.videolib import *

if __name__ == "__main__":
    
    all_frames = LocalVideoToFrames("VIRAT_S_000200_02_000479_000635.mp4")()
    sampled_frames = Slice(step=30)(all_frames)
    detections = Detection('cloudlet015.elijah.cs.cmu.edu', 5000)(sampled_frames)
    person_detections = DetectionFilter(['person'], 0.5)(detections)

    for k, intrvl in enumerate(run_to_finish(person_detections)):
        print(intrvl['t1'], intrvl['t2'])
        intrvl.savefile(f"output_person_frame/{k}.jpg")
