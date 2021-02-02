from stsearch.videolib import LocalVideoDecoder


def query(path, session):
    decoder = LocalVideoDecoder(path)
    frame_count = decoder.frame_count
    fps = decoder.fps
    H, W = decoder.raw_height, decoder.raw_width

    return {
        'frame_count': frame_count,
        'fps': fps,
        'height': H,
        'width': W
    }



def main(output="okutama_metadata.csv"):
    from pathlib import Path
    import pickle
    import pandas as pd
    from stsearch.diamond_wrap.result_pb2 import STSearchResult
    from stsearch.diamond_wrap.utils import start_stsearch_by_script, OUTPUT_ATTR

    results = start_stsearch_by_script(open(__file__, 'rb').read())

    total_frames = 0
    total_hrs = 0
    save_results = []


    for i, res in enumerate(results):
        object_id = res['_ObjectID'].decode()
        clip_id = Path(object_id).stem

        filter_result: STSearchResult = STSearchResult()
        filter_result.ParseFromString(res[OUTPUT_ATTR])
        query_result = pickle.loads(filter_result.query_result)
        print(f"{clip_id}, {query_result}")
        total_frames += query_result['frame_count']
        total_hrs += query_result['frame_count'] / query_result['fps'] / 3600

        save_results.append(
            {
                'clip_id': clip_id,
                'frame_count': query_result['frame_count'],
                'fps': query_result['fps'],
                'height': query_result['height'],
                'width': query_result['width']
            }
        )

        pd.DataFrame(save_results).to_csv(output)

    print(f"total_hrs: {total_hrs}. Total frames: {total_frames}. Count: {i}")


if __name__ == '__main__':
    import fire
    fire.Fire(main)