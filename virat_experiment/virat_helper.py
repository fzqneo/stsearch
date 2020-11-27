import os
from pathlib import Path

import pandas as pd

VIRAT_ROOT = "/home/zf/VIRAT/VIRAT/Public Dataset/VIRAT Video Dataset Release 2.0/VIRAT Ground Dataset/"

LIST_CLIP_ID = [p.stem for p in (Path(VIRAT_ROOT) / 'videos_original').iterdir()]
assert len(LIST_CLIP_ID) == 329

class EVENTTYPE(object):
    LOAD_OBJECT_VEHICLE=1
    UNLOAD_OBJECT_VEHICLE=2
    OPEN_TRUNK=3
    CLOSE_TRUNK=4
    GET_IN_VEHICLE=5
    GET_OUT_VEHICLE=6
    GESTURE=7
    DIG=8
    CARRY_OBJECT=9
    RUN=10
    ENTER_FACILITY=11
    EXIT_FACILITY=12

def load_events(clip_id):
    columns = [
        'event_id', 'event_type', 'duration', 'start_frame', 'end_frame',
        'current_frame', 'x1', 'y1', 'w', 'h'
    ]
    path = Path(VIRAT_ROOT) / "annotations" / f"{clip_id}.viratdata.events.txt"
    df = pd.read_csv(path, sep=' ', names=columns, index_col=False)
    return df


def load_summary():
    path = Path(__file__).parent / "README_annotations_evaluations.csv"
    df = pd.read_csv(path, sep=',', header=0, index_col=False)
    return df


if __name__ == "__main__":
    # df = load_events("VIRAT_S_010203_02_000347_000397")
    df = load_summary()
    print(df)