import os
import pickle
import tempfile

from opendiamond.filter import Filter, Session
from opendiamond.filter.parameters import StringParameter

def get_query_fn(blob):
    d = {}
    exec(blob, d)
    return d['query']

class STSearchFilter(Filter):

    params = (
        StringParameter('output_attr'),
    )

    blob_is_zip = False

    def __init__(self, args, blob, session=Session('filter')):
        super().__init__(args, blob, session)

        # load query function from blob
        query_fn = get_query_fn(self.blob)
        assert callable(query_fn)
        self.query_fn = query_fn

    def __call__(self, obj):

        # save obj to tempfile
        f = tempfile.NamedTemporaryFile('wb', suffix='.mp4', prefix='STSearchFilter', delete=False)
        f.write(obj.data)
        f.close()

        # init and execute query, buffer all results
        query_result = self.query_fn(f.name)

        # delete tempfile
        os.unlink(f.name)

        # set results as attribute
        # Shall we let the user create the attribute binary?
        # or do we assume it must be a list of Interval?
        obj.set_binary(self.output_attr, pickle.dumps(query_result))
        return True