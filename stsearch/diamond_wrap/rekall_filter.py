from opendiamond.filter import Filter, Session
from opendiamond.filter.parameters import *


def get_query_fn(blob):
    d = {}
    exec(blob, d)
    return d['query']

class RekallFilter(Filter):

    blob_is_zip = False

    def __init__(self, args, blob, session=Session('filter')):
        super().__init__(args, blob, session)

        # load query function from blob
        query_fn = get_query_fn(self.blob)
        assert callable(query_fn)
        self.query_fn = query_fn


    def __call__(self, obj):

        # save obj to tempfile

        # init and execute query, buffer all results

        # delete tempfile

        # set results as attribute
        
        return True