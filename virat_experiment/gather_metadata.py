from pathlib import Path

from stsearch.videolib import LocalVideoDecoder


def query(path, session):
    decoder = LocalVideoDecoder(path)

    rv = {
        'frame_count': decoder.frame_count,
        'fps': decoder.fps,
        'width': decoder.raw_width,
        'height': decoder.raw_height,
        'filesize': Path(path).stat().st_size
    }

    return rv


if __name__ == "__main__":
    from pathlib import Path
    import pickle
    from stsearch.diamond_wrap.result_pb2 import STSearchResult
    from utils import start_stsearch_by_script, OUTPUT_ATTR

    import pandas as pd

    results = start_stsearch_by_script(open(__file__, 'rb').read())
    all_metadata = []

    for i, res in enumerate(results):
        object_id = res['_ObjectID'].decode()
        clip_id = Path(object_id).stem

        filter_result: STSearchResult = STSearchResult()
        filter_result.ParseFromString(res[OUTPUT_ATTR])
        query_result = pickle.loads(filter_result.query_result)
        print(f"{clip_id}, {filter_result.stats}, {query_result}")

        query_result['clip_id'] = clip_id
        all_metadata.append(query_result)

    df = pd.DataFrame(all_metadata)
    df.to_csv("VIRAT_metadata.csv")




