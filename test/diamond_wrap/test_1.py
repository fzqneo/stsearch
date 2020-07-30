import pickle
import os
import pathlib
from typing import Iterable

import cv2
from logzero import logger

from opendiamond.client.search import Blob, DiamondSearch, FilterSpec
from opendiamond.client.util import get_default_scopecookies, create_filter_from_files

def get_default_code_path(code_file):
    code_path = os.path.join(os.environ['HOME'], '.diamond', 'filters', code_file)
    return code_path

OUTPUT_ATTR = 'stsearch-output'

SCRIPT_FILE = "script_1st_frame.py"
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
        result = pickle.loads(res[OUTPUT_ATTR])
        print(i, object_id, type(result))
        # cv2.imwrite(f"{pathlib.Path(object_id).stem}.jpg", result)
        with open(f"{pathlib.Path(object_id).stem}.jpg", 'wb') as f:
            f.write(result)

    stats = search.get_stats()
    search.close()
    print(stats)


