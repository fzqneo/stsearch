"""This modules defines classes that conform to `opendiamond.filter.Filter`.
It intends to provide STSearch functionality in an existing OpenDiamond system.
"""

import io
import os
import pickle
import tempfile
import time
import zipfile

from opendiamond.filter import Filter, Session
from opendiamond.filter.parameters import StringParameter

from stsearch.diamond_wrap.result_pb2 import STSearchResult


def get_query_fn(blob):
    d = {}
    exec(blob, d)
    return d['query']

class STSearchFilter(Filter):

    params = (
        StringParameter('output_attr'),
        StringParameter('mode'), # 'script' or 'zip'
    )

    blob_is_zip = False

    def __init__(self, args, blob, session=Session('filter')):
        super().__init__(args, blob, session)

        self.scriptdir = None

        if self.mode == 'zip':
            # the zip mode have the side effect of changing cwd of the current processs
            
            # create temp dir and cwd to it
            self.scriptdir = tempfile.TemporaryDirectory(prefix='stsearch-filter-zip-mode')
            os.chdir(self.scriptdir.name)
            
            # extract the blob zip in it
            with zipfile.ZipFile(io.BytesIO(self.blob)) as zf:
                zf.extractall()

            # load query function from script.py, which possibly opens local paths 
            with open('script.py', 'rb') as qf:
                query_fn = get_query_fn(qf.read())
                assert callable(query_fn)
                self.query_fn = query_fn

        else:   # script mode
            # load query function from blob
            query_fn = get_query_fn(self.blob)
            assert callable(query_fn)
            self.query_fn = query_fn

    def __call__(self, obj):
        # get obj data
        tic = time.time()
        _ = obj.data
        ipc_time = time.time() - tic

        # save obj to tempfile
        tic = time.time()
        f = tempfile.NamedTemporaryFile('wb', suffix='.mp4', prefix='STSearchFilter', delete=False)
        f.write(obj.data)
        f.close()
        save_time = time.time() - tic

        # init and execute query, buffer all results
        tic = time.time()
        query_result = self.query_fn(f.name, session=self.session)
        query_time = time.time() - tic

        # delete tempfile
        os.unlink(f.name)

        tic = time.time()
        query_result_serialized = pickle.dumps(query_result)
        pickle_time = time.time() - tic

        msg = STSearchResult()
        msg.query_result = query_result_serialized
        msg.stats.update({
            'input_size': float(len(obj.data)),
            'ipc_time': ipc_time,
            'save_time': save_time,
            'query_time': query_time,
            'pickle_time': pickle_time
        })

        obj.set_binary(self.output_attr, msg.SerializeToString())
        return True

    def __del__(self):
        if self.scriptdir:
            self.scriptdir.cleanup()