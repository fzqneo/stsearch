import pickle
import os
import pathlib

import cv2
from logzero import logger

from opendiamond.client.search import Blob, DiamondSearch, FilterSpec
from opendiamond.client.util import get_default_scopecookies, create_filter_from_files

from stsearch.diamond_wrap.result_pb2 import STSearchResult

def get_default_code_path(code_file):
    code_path = os.path.join(os.environ['HOME'], '.diamond', 'filters', code_file)
    return code_path

OUTPUT_ATTR = 'stsearch-output'

SCRIPT_FILE = "script.py"
SCRIPT_CONTENT = open(SCRIPT_FILE, 'rb').read()

# SCRIPT_CONTENT = """
# import os 
# def query(path):
#     return f"This runner gets path: {path},  size {os.path.getsize(path)}"
# """

if __name__ == "__main__":
    script_blob = Blob(SCRIPT_CONTENT)
    code_blob = Blob(open(get_default_code_path("fil_stsearch.py"), 'rb').read())
    fil_stsearch_spec = FilterSpec(
        name="fil-stsearch",
        code=code_blob,
        arguments=(OUTPUT_ATTR, 'script'),
        blob_argument=script_blob
    )

    search = DiamondSearch(get_default_scopecookies(), [fil_stsearch_spec,], push_attrs=[OUTPUT_ATTR,])
    search_id = search.start()
    for i, res in enumerate(search.results):
        object_id = res['_ObjectID'].decode()

        filter_result = STSearchResult()
        filter_result.ParseFromString(res[OUTPUT_ATTR])
        query_result = pickle.loads(filter_result.query_result)
        print(f"{object_id}, {filter_result.stats}, {len(query_result)}")

        for k, (bound, blob, ext) in enumerate(query_result):
            # print(object_id, k, bound, len(blob), ext)
            with open(f"{pathlib.Path(object_id).stem}-{k}.{ext}", 'wb') as f:
                f.write(blob)

    stats = search.get_stats()
    search.close()
    print(stats)


