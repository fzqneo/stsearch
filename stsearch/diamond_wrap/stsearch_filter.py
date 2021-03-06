"""This modules defines classes that conform to `opendiamond.filter.Filter`.
It intends to provide STSearch functionality in an existing OpenDiamond system.
"""

import gc
import io
import multiprocessing
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

        self.scriptdir = tempfile.TemporaryDirectory(prefix='stsearch-script-')

        if self.mode == 'zip':
            # the zip mode have the side effect of changing cwd of the current processs            
            os.chdir(self.scriptdir.name)
            self.session.log('info', f"Zip mode running at {self.scriptdir.name}")
            
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
            # just for debug
            with tempfile.NamedTemporaryFile(prefix="autosave-script-", suffix=".py", dir=self.scriptdir.name, delete=False) as f1:
                f1.write(self.blob)

    def __call__(self, obj):
        gc.collect()
        # get obj data
        tic = time.time()
        _ = obj.data
        ipc_time = time.time() - tic

        # save obj to tempfile
        tic = time.time()
        tmpf = tempfile.NamedTemporaryFile('wb', suffix='.mp4', prefix='STSearchFilter-', delete=False)
        tmpf.write(obj.data)
        tmpf.close()
        save_time = time.time() - tic


        # init and execute query, buffer all results
        tic = time.time()
        self.session.log('info', f"starting query() working on {tmpf.name}")

        # use mp to workaround memory leaks in OpenCV
        def child_f(path, query_fn, conn):
            conn.send(query_fn(path, None))
            conn.close()

        parent_conn, child_conn = multiprocessing.Pipe()
        p = multiprocessing.Process(target=child_f, args=(tmpf.name, self.query_fn, child_conn))
        p.start()
        query_result = parent_conn.recv()
        p.join()
        # query_result = self.query_fn(tmpf.name, session=self.session)
        self.session.log('info', f"query() done on {tmpf.name}")
        query_time = time.time() - tic

        # delete tempfile
        os.unlink(tmpf.name)

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