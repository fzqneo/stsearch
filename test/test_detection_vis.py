from stsearch.cvlib import Detection, DetectionFilter, DetectionVisualize
from stsearch.interval import *
from stsearch.op import *
from stsearch.utils import run_to_finish
from stsearch.videolib import *

if __name__ == "__main__":
    
    path = "example.mp4"
    all_frames = VideoToFrames(LocalVideoDecoder(path))()

    sampled_frames = Slice(step=15)(all_frames)
    detections = Detection('cloudlet015.elijah.cs.cmu.edu', 5000)(sampled_frames)
    vis_frames = DetectionVisualize(['person'], 0.3)(detections)

    output = vis_frames
    output_sub = output.subscribe()
    output.start_thread_recursive()

    for k, intrvl in enumerate(output_sub):
        print(intrvl['t1'], intrvl['t2'])
        intrvl.savefile(f"output_detection_vis/{k}-{intrvl['t1']:05d}-{intrvl['t2']:05d}.jpg")
