import pickle
import os
import pathlib

import cv2
from logzero import logger

from opendiamond.client.search import Blob, DiamondSearch, FilterSpec
from opendiamond.client.util import get_default_scopecookies, create_filter_from_files

def get_default_code_path(code_file):
    code_path = os.path.join(os.environ['HOME'], '.diamond', 'filters', code_file)
    return code_path

OUTPUT_ATTR = 'stsearch-output'

SCRIPT_FILE = "script_1.py"
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
        arguments=(OUTPUT_ATTR, ),
        blob_argument=script_blob
    )

    search = DiamondSearch(get_default_scopecookies(), [fil_stsearch_spec,], push_attrs=[OUTPUT_ATTR,])
    search_id = search.start()
    for i, res in enumerate(search.results):
        object_id = res['_ObjectID'].decode()
        results = pickle.loads(res[OUTPUT_ATTR])
        print(object_id, len(results))
        for k, (bound, blob, ext) in enumerate(results):
            print(object_id, k, bound, len(blob), ext)
            with open(f"{pathlib.Path(object_id).stem}-{k}.{ext}", 'wb') as f:
                f.write(blob)

    stats = search.get_stats()
    search.close()
    print(stats)


