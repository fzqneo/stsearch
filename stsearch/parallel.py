import concurrent.futures
import typing
import uuid

from stsearch import Interval
from stsearch.op import Flatten, Graph, Map, Op

class ParallelMap(Graph):
    """Use `concurruent.futures.ThreadPoolExecutor` to execute the map on several
    inputs in parallel, maintaining their output order. 
    This is implemented by two vanilla `Map`.

    Args:
        Graph ([type]): [description]
    """

    def __init__(self, map_fn, name=None, max_workers=None, executor=None):
        super().__init__()
        self.map_fn = map_fn
        self.name = name
        self.max_workers = max_workers
        self.executor = executor

    def call(self, instream):
        executor = self.executor or concurrent.futures.ThreadPoolExecutor(self.max_workers)
        future_key = 'PMapKey' + str(uuid.uuid4())

        def dispatch_future(i1: Interval) -> Interval:
            fut = executor.submit(self.map_fn, i1)
            return Interval(
                i1.bounds,
                {future_key: fut}
            )

        def collect_future(i1: Interval) -> Interval:
            fut = i1.payload[future_key]
            return fut.result()

        futures = Map(
            map_fn=dispatch_future,
            name=f"{self.__class__.__name__}/dispatch/{future_key}:{self.name}"
        )(instream)

        return Map(
            map_fn=collect_future,
            name=f"{self.__class__.__name__}/collect/{future_key}:{self.name}"
        )(futures)

class ParallelFlatten(Graph):

    def __init__(
        self, 
        flatten_fn: typing.Callable[[Interval], typing.List[Interval]], 
        name=None,
        max_workers=None,
        executor=None):

        self.flatten_fn = flatten_fn
        self.name = name
        self.max_workers = max_workers
        self.executor = executor

    def call(self, instream):
        key = 'PFlattenKey'

        def wrap_map_fn(i1: Interval) -> Interval:
            rv: typing.List[Interval] = self.flatten_fn(i1)
            return Interval(i1.bounds, {key: rv})

        def wrap_flatten_fn(i1: Interval) -> typing.List[Interval]:
            return i1.payload[key]

        wrapped_result = ParallelMap(
            wrap_map_fn,
            name=f"{self.__class__.__name__}/PMap/{self.name}",
            max_workers=self.max_workers,
            executor=self.executor
        )(instream)

        return Flatten(
            wrap_flatten_fn,
            name=f"{self.__class__.__name__}/Flatten/{self.name}"
        )(wrapped_result)


# TODO: ParallelFilter can be implemented with Map in similar way.