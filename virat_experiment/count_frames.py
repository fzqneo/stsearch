from stsearch.videolib import LocalVideoDecoder


def query(path, session):
    decoder = LocalVideoDecoder(path)
    frame_count = decoder.frame_count
    fps = decoder.fps

    return {
        'frame_count': frame_count,
        'fps': fps
    }



if __name__ == "__main__":
    from pathlib import Path
    import pickle
    from stsearch.diamond_wrap.result_pb2 import STSearchResult
    from utils import start_stsearch_by_script, OUTPUT_ATTR

    results = start_stsearch_by_script(open(__file__, 'rb').read())

    total_frames = 0
    total_hrs = 0


    for i, res in enumerate(results):
        object_id = res['_ObjectID'].decode()
        clip_id = Path(object_id).stem

        filter_result: STSearchResult = STSearchResult()
        filter_result.ParseFromString(res[OUTPUT_ATTR])
        query_result = pickle.loads(filter_result.query_result)
        print(f"{clip_id}, {filter_result.stats}, {query_result}")
        total_frames += query_result['frame_count']
        total_hrs += query_result['frame_count'] / query_result['fps'] / 3600

    print(f"total_hrs: {total_hrs}. Total frames: {total_frames}. Count: {i}")


