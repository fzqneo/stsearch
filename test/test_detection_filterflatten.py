from stsearch.cvlib import Detection, DetectionFilter, DetectionFilterFlatten
from stsearch.interval import *
from stsearch.op import *
from stsearch.utils import run_to_finish
from stsearch.videolib import *

if __name__ == "__main__":
    
    all_frames = LocalVideoToFrames("FifthCraig1-2019-02-01-10-05-05.mp4")()
    sampled_frames = Slice(step=300)(all_frames)
    detections = Detection('cloudlet015.elijah.cs.cmu.edu', 5000)(sampled_frames)
    person_detections = DetectionFilterFlatten(['person'], 0.5)(detections)

    for k, intrvl in enumerate(run_to_finish(person_detections)):
        print(intrvl['t1'], intrvl['t2'])
        intrvl.savefile(f"output_detection_filterflatten/{k}-{intrvl['t1']}-{intrvl['t2']}.jpg")
