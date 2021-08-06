import hashlib
import json
import os
from pathlib import Path

import fire
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


def parse_event_list(clip_id, event_type) -> pd.DataFrame:
    # compress each event instance into a single row
    try:
        df = load_events(clip_id)
    except FileNotFoundError:
        # some clips have no events and no event file
        return pd.DataFrame()
    
    df = df[df['event_type']==event_type]
    df['x2'] = df['x1'] + df['w']
    df['y2'] = df['y1'] + df['h']
    df['t1'], df['t2'] = df['start_frame'], df['end_frame']
    
    rv = []
    for event_id in df['event_id'].unique():
        edf = df[df['event_id']==event_id]
        rv.append({
            'event_id': event_id,
            'x1': edf['x1'].min(),
            'x2': edf['x2'].max(),
            'y1': edf['y1'].min(),
            'y2': edf['y2'].max(),
            't1': edf['t1'].min(),
            't2': edf['t2'].max(),
        })
        
    return  pd.DataFrame(rv)

def parse_all_event_list(event_type) -> pd.DataFrame:
    all_df = []
    for clip_id in LIST_CLIP_ID:
        df = parse_event_list(clip_id, event_type)
        if not df.empty:
            df['clip_id'] = clip_id
            all_df.append(df)
        
    return pd.concat(all_df, ignore_index=True)


def parse_result(result_file):
    df = pd.read_csv(result_file, index_col=0)
    # convert relative coord to pixel coord
    df['x1'] = df['x1'] * df['width']
    df['x2'] = df['x2'] * df['width']
    df['y1'] = df['y1'] * df['height']
    df['y2'] = df['y2'] * df['height']
    return df


class OBJECTTYPE(object):
    PERSON=1
    CAR=2
    VEHICLE=3
    OBJECT=4
    BIKE=5


def load_objects(clip_id):
    columns = [
        'track_id', 'duration', 'current_frame', 'x1', 'y1', 'w', 'h', 'object_type'
    ]
    path = Path(VIRAT_ROOT) / "annotations" / f"{clip_id}.viratdata.objects.txt"
    try:
        df = pd.read_csv(path, sep=' ', names=columns, index_col=False)
        df['x2'] = df['x1'] + df['w']
        df['y2'] = df['y1'] + df['h']
        return df
    except FileNotFoundError:
        print(f"No object file for {clip_id}. Returning empty DataFrame")
        return pd.DataFrame(columns=columns)


def convert_gt_and_det_for_metrics(gt:str, det:str, cache_dir="cache", classes=["car", "bus", "truck"]):
    # convert predictions and gt into files used by https://github.com/rafaelpadilla/Object-Detection-Metrics
    for cache_file in Path(cache_dir).glob("*.json"):
        with open(cache_file, 'rt') as f:
            cache_result = json.load(f)

        clip_id = cache_result['clip_id']

        print(f"Found {cache_file}. Clip id={clip_id}")

        fps = cache_result['fps']
        W, H = cache_result['width'], cache_result['height']

        frame_ids = []  # frame ids that have been detected

        # write the det file
        for frame_result in cache_result['detection']:
            frame_id = frame_result['t1']
            frame_ids.append(frame_id)
            D = frame_result['detection']
            with open(Path(det)/f"{clip_id}_{frame_id}.txt", "wt") as of:
                for class_name, score, (top, left, bottom, right) in zip(D['detection_names'], D['detection_scores'], D['detection_boxes']):
                    if class_name in classes and score > 0.:
                        of.write(f"vehicle {score} {left*W} {top*H} {right*W} {bottom*H}\n")

        # write the groundtruth file using frame_ids
        df = load_objects(clip_id)
        for frame_id in frame_ids:
            with open(Path(gt)/f"{clip_id}_{frame_id}.txt", "wt") as of:
                for _, row in df[df['current_frame']==frame_id].iterrows():
                    if row['object_type'] in [OBJECTTYPE.CAR, OBJECTTYPE.VEHICLE]:
                        of.write(f"vehicle {row['x1']} {row['y1']} {row['x2']} {row['y2']}\n")
                
                
if __name__ == "__main__":
    fire.Fire()    