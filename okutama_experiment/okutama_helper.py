import hashlib
import json
from pathlib import Path

import fire
import pandas as pd

OKUTAMA_ROOT = "/home/zf/okutama/Train-Set"

OKUTAMA_LABEL_DIR = str(Path(OKUTAMA_ROOT) / "Labels/SingleActionTrackingLabels/3840x2160")
LIST_CLIP_ID = [p.stem for p in Path(OKUTAMA_LABEL_DIR).iterdir() if p.suffix=='.txt']
# print(LIST_CLIP_ID)
assert len(LIST_CLIP_ID) == 33


def load_label_file(clip_id:str) -> pd.DataFrame:
    path = Path(OKUTAMA_LABEL_DIR) / f"{clip_id}.txt"
    columns = ['track_id', 'x1', 'y1', 'x2', 'y2', 'frame_id', 'lost', 'occluded', 'generated', 'label', 'action']
    df = pd.read_csv(path, sep=' ', names=columns, index_col=False)
    
    # remove occluded and lost
    df = df[(df['lost']==0) & (df['occluded']==0)]
    
    return df

def compress_event(clip_id:str, event:str) -> pd.DataFrame:
    # compress an event instance into a single row

    df = load_label_file(clip_id)

    # select event
    df = df[df['action'] == event]
    
    # compress track into one record
    rv = []
    for track_id in df['track_id'].unique():
        v = df[df['track_id']==track_id]
        rv.append(
        {
            'track_id': track_id,
            'x1': v['x1'].min(),
            'x2': v['x2'].max(),
            'y1': v['y1'].min(),
            'y2': v['y2'].max(),
            't1': v['frame_id'].min(),
            't2': v['frame_id'].max()
        })
    
    return pd.DataFrame(rv)


def compress_event_all_clips(event:str) -> pd.DataFrame:
    all_df = []
    for clip_id in LIST_CLIP_ID:
        df = compress_event(clip_id, event)
        if not df.empty:
            df['clip_id'] = clip_id
            all_df.append(df)
    return pd.concat(all_df, ignore_index=True)

def convert_gt_and_det_for_metrics(gt:str, det:str, cache_dir="okutama_cache", classes=["person",]):
    for cache_file in Path(cache_dir).glob("*.json"):
        with open(cache_file, 'rt') as f:
            cache_result = json.load(f)

        clip_id = cache_result['clip_id']
        frame_count = cache_result['frame_count']
        W, H = cache_result['width'], cache_result['height']

        print(f"Found {cache_file}. Clip_id={clip_id}")

        # write the det files
        for frame_id in range(frame_count):
            with open(Path(det)/f"{clip_id}_{frame_id}.txt", 'wt') as of:
                for r in filter(lambda r:r['current_frame']==frame_id, cache_result['detections']):
                    if r['score'] > 0. and r['class_name'] in classes:
                        of.write(f"person {r['score']} {r['x1']*W} {r['y1']*H} {r['x2']*W} {r['y2']*H}\n")

        # write the gt files
        df = load_label_file(clip_id)
        for frame_id in range(frame_count):
            with open(Path(gt)/f"{clip_id}_{frame_id}.txt", 'wt') as of:
                for _, row in df[df['frame_id']==frame_id].iterrows():
                    of.write(f"person {row['x1']} {row['y1']} {row['x2']} {row['y2']}\n")

        
if __name__ == "__main__":
    fire.Fire()    
