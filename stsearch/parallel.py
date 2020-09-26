import concurrent.futures
import uuid

from stsearch import Interval
from stsearch.op import Graph, Map, Op

class ParallelMap(Graph):

    def __init__(self, map_fn, name=None, max_workers=None, executor=None):
        super().__init__()
        self.map_fn = map_fn
        self.name = name
        self.max_workers = max_workers
        self.executor = executor

    def call(self, instream):
        executor = self.executor or concurrent.futures.ThreadPoolExecutor(self.max_workers)
        future_key = 'PMapKey' + str(uuid.uuid4())

        def generate_future(i1: Interval) -> Interval:
            fut = executor.submit(self.map_fn, i1)
            return Interval(
                i1.bounds,
                {future_key: fut}
            )

        def consume_future(i1: Interval) -> Interval:
            fut = i1.payload[future_key]
            return fut.result()

        futures = Map(
            map_fn=generate_future,
            name=f"{self.__class__.__name__}/generate/{future_key}:{self.name}"
        )(instream)

        return Map(
            map_fn=consume_future,
            name=f"{self.__class__.__name__}/consume/{future_key}:{self.name}"
        )(futures)


# TODO: ParallelFilter can be implemented with Map in similar way.