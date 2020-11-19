import os
import typing


from opendiamond.client.search import Blob, DiamondSearch, FilterSpec
from opendiamond.client.util import get_default_scopecookies


OUTPUT_ATTR = 'stsearch-output'

def start_stsearch_by_script(scrit_content: bytes) -> typing.Iterator:
    code_path = os.path.join(os.environ['HOME'], '.diamond', 'filters', 'fil_stsearch.py')
    code_blob = Blob(open(code_path, 'rb').read())
    
    script_blob = Blob(scrit_content)

    fil_stsearch_spec = FilterSpec(
        name="fil-stsearch",
        code=code_blob,
        arguments=(OUTPUT_ATTR, 'script'),
        blob_argument=script_blob
    )

    search = DiamondSearch(get_default_scopecookies(), [fil_stsearch_spec,], push_attrs=[OUTPUT_ATTR,])
    _ = search.start()
    return search.results