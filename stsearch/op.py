import collections
import functools
import heapq
from logzero import logger
import threading
import typing
import uuid

import rekall
from rekall.bounds import Bounds, Bounds3D
from rekall.predicates import *

from stsearch.interval import Interval
from stsearch.invertal_stream import IntervalStream, IntervalStreamSubscriber

class Graph(object):
    """
    ``Graph`` is used to compose a (sub)-graph using ``Op``s.
    ``Graph`` has a call() method similar to that of ``Op``, but doesn't have ``execute()``
    and ``publish()``. It doesn't create its own output stream and doesn't have its own thread.
    """

    def __init__(self):
        super().__init__()

    def call(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        # traverse all args, check type
        for a in list(args) + list(kwargs.values()):
            assert isinstance(a, IntervalStream), "Should only pass IntervalStream into call()"

        return self.call(*args, **kwargs)


class Op(object):
    """An operator takes one or multiple ``IntervalStream`` as input and outputs one ``IntervalStream``.

    The design of ``Op`` is inspired by both data flow systems such as TensorFlow and 
    relational databases such as PostgreSQL.
    A subclass of ``Op`` should implement the ``call()`` method and the ``execute()`` method.

    The ``call()`` method is similar to that of Kera's Layer's. It accepts one or multiple 
    ``IntervalStreamSubscriber`` as arguments, which allows the author to do pre-processing
    and create references to be used in ``execute()``.
    The user calls this method indirectly by calling the builtin ``__call__`` method, which accepts
    one or multiple ``IntervalStream`` and return one ``IntervalStream``.
    Internally, it firstly creates ``IntervalStreamSubscriber``s to the corresponding ``IntervalStream``s 
    and passes those to ``call()``; it then creates an output ``IntervalStream`` of this Op.

    >>> ouput_stream = some_op_class(op_param1, op_param_2)(input_stream_1, input_stream_2)
    
    The ``execute()`` method is called, typically repeatedly, to consume the input streams and publish results 
    to the output stream. Each call of ``execute()`` has this semantics: if it returns ``True``, it means
    it has written at least one result to the output stream (maybe more); if it returns ``False``, it means
    the input streams have been exhausted and no more results will be output. Therefore, a typicall 
    implementation of an Op's ``execute()`` will keep consuming its input stream(s) until it is able to
    produce at least one output or exhausts the input, and then returns.
    
    """

    # support a default name
    _name_counter = collections.Counter()
    
    def __init__(self, name: typing.Optional[str] = None):
        super().__init__()
        self._inputs = None
        self.output = None
        self.started = False

        if name is not None:
            self.name = name
        else:
            self.name = self.__class__.__name__ + '-' + str(Op._name_counter[self.__class__.__name__])
            Op._name_counter[self.__class__.__name__] += 1

    def call(self, *args: IntervalStreamSubscriber, **kwargs: IntervalStreamSubscriber):
        raise NotImplementedError

    def __call__(self, *args: IntervalStream, **kwargs: IntervalStream) -> IntervalStream:
        assert self.output is None, "call() has been called already. We don't support reusing the same Op on a different set of input stream(s) yet"
        inputs = list(args) + list(kwargs.values())
        # traverse all args, check type, and convert them into ``IntervalStreamSubscriber`` of the corresponding ``IntervalStream``
        for a in inputs:
            assert isinstance(a, IntervalStream), f"Should only pass IntervalStream into call(), got {type(a)}"
        self._inputs = inputs

        args_sub = [a.subscribe() for a in args]
        kwargs_sub = dict([(k, v.sbuscribe()) for k, v in kwargs.items()])

        self.call(*args_sub, **kwargs_sub)
        # create an output stream
        self.output = IntervalStream(parent=self)
        return self.output

    def execute(self) -> bool:
        raise NotImplementedError

    def loop_execute(self):
        while self.execute():
            pass
        self.publish(None)

    def start_thread(self):
        if not self.started:
            t = threading.Thread(target=self.loop_execute, name=f"op-thread-{self.name}", daemon=True)
            t.start()
            logger.debug(f"Started operator thread {t.name}")
            self.started = True

    def start_thread_recursive(self):
        for istream in self._inputs:
            istream.parent.start_thread_recursive()

        self.start_thread()

    def publish(self, i: Interval):
        assert i is None or isinstance(i, Interval), f"Expect None or Interval. Got {type(i)}"
        assert self.output is not None, "call() has not been called"
        self.output.publish(i)


class Slice(Op):

    def __init__(self, start=0, end=None, step=1, name=None):
        super().__init__(name=name)
        self.start = start
        self.end = end
        self.step = step
        self.ind = 0

    def call(self, instream):
        self.instream = instream

    def execute(self):
        while True:
            itvl = self.instream.get()
            if not itvl:
                return False

            try:
                if self.ind >= self.start and (not self.end or self.ind < self.end) and (self.ind - self.start) % self.step == 0:
                    self.publish(itvl)
                    return True
                elif self.end is not None and self.ind >= self.end: # pass send
                    return False
                else:   # skip
                    pass
            finally:
                self.ind += 1


class Map(Op):
    def __init__(self, map_fn: typing.Callable[[Interval], Interval], name=None):
        super().__init__(name=name)
        self.map_fn = map_fn

    def call(self, instream):
        self.instream = instream

    def execute(self):
        i = self.instream.get()
        if i is not None:
            self.publish(self.map_fn(i))
            return True
        else:
            return False


class Filter(Op):
    def __init__(self, pred_fn: typing.Callable[[Interval], bool], name=None):
        super().__init__(name)
        self.pred_fn = pred_fn

    def call(self, instream):
        self.instream = instream

    def execute(self):
        while True:
            i = self.instream.get()
            if i is None:
                return False
            elif self.pred_fn(i):
                self.publish(i)
                return True
            else:
                continue


class FromIterable(Op):
    def __init__(self, iterable_of_intervals: typing.Iterable[Interval], name=None):
        super().__init__(name)
        self.iterable_of_intervals = iterable_of_intervals
        self.iterator = iter(iterable_of_intervals)

    def call(self):
        pass

    def execute(self):
        try:
            i = next(self.iterator)
            self.publish(i)
            return True
        except StopIteration:
            return False


class SetOp(Op):
    """This encapsulates Rekall-style batch operators that work on finite 
    ``IntervalSet`` rather than streams.
    Internally, it first buffers all input ``Interval``, creates ``IntervalSet``
    containing them, and then invokes the passed-in ``setop_fn``. The ``setop_fn``
    accepts one or multiple ``IntervalSet`` and returns one ``IntervalSet``. 
    The resulting ``IntervalSet`` is then buffered and published in subsequent
     ``execute()`` calls.

    Because of its set semantics, the first call to ``execute()`` will block
    until all upstream Ops have finished. Hence, this Op should not be used on 
    streams, because they are endless. There can also be unintended memory issues 
    when the payload of input/output intervals are huge.

    The primary purpose of this Op is provide an easy way to reuse the functions
    provided by Rekall.
    """

    def __init__(self, setop_fn: typing.Callable[..., rekall.IntervalSet], name=None):
        super().__init__(name)
        self.setop_fn = setop_fn
        self.result_buffer = None
        self.done = False

    def call(self, *args):
        self.instream_subs = args

    def execute(self):
        if not self.done:
            logger.debug(f"Setop {self.name}: gathering input sets")
            # buffer all input and perform setop
            intrvl_set_args = [rekall.IntervalSet(list(iter(sub))) for sub in self.instream_subs]
            logger.debug(f"Setop {self.name}: input set sizes: {list(map(len, intrvl_set_args))}")
            result_intrvl_set = self.setop_fn(*intrvl_set_args)
            assert isinstance(result_intrvl_set, rekall.IntervalSet)
            self.result_buffer = sorted(result_intrvl_set._intrvls)
            self.done = True

        if len(self.result_buffer) > 0:
            ret = self.result_buffer.pop(0)
            # type cast
            ret = Interval(ret.bounds, ret.payload)
            self.publish(ret)
            return True
        else:
            return False


class SetMap(Graph):

    def __init__(self, map_fn, name=None):
        super().__init__()
        self.name = name
        self.map_fn = map_fn

    def call(self, instream):
        return SetOp(
            setop_fn=functools.partial(rekall.IntervalSet.map, map_fn=self.map_fn),
            name=self.name
        )(instream)


class SetCoalesce(Graph):

    def __init__(self, **kwargs):
        self.coalesce_kwargs = kwargs

    def call(self, instream):
        return SetOp(
            setop_fn=functools.partial(rekall.IntervalSet.coalesce, **self.coalesce_kwargs)
        )(instream)


class Coalesce(Op):
    """The stream version of coalesce. Due to the streaming nature, pass ``axis=`` anything other
    than ``('t1', 't2')`` doesn't really make sense.

    """

    def __init__(self, 
                bounds_merge_op: typing.Callable[[Bounds, Bounds], Bounds] = Bounds3D.span,
                payload_merge_op: typing.Callable[[dict, dict], dict] = lambda p1, p2: p1,
                predicate: typing.Optional[typing.Callable[[Interval, Interval], bool]] = None,
                epsilon: float = 0.,
                axis=('t1', 't2'),
                interval_merge_op: typing.Optional[typing.Callable[[Interval, Interval], Interval]] = None,
                name=None):
        """If ``interval_merge_op`` is not None, ``payload_merge_op`` and ``bounds_merge_op`` will be ignored.

        Args:
            bounds_merge_op (function): takes two ``Bounds`` objects and returns one ``Bounds`` object
            payload_merge_op (function, optional): takes two payloads and returns one. Defaults to taking the first payload.
            predicate (function, optional): takes two intervals and return Boolean. Defaults to None.
            epsilon (int, optional): [description]. Defaults to 0.
            axis (tuple, optional): [description]. Defaults to ('t1', 't2').
            interval_merge_op (function, optional): takes two ``Interval`` objects and returns one. 
                If not None, it is used to create a merged interval; ``bounds_merge_op`` and ``payload_merge_op`` will be
                ignored in this case. This is useful, e.g., when the new payload depends on the bounds of the input intervals.
                Defaults to None.
            name ([type], optional): [description]. Defaults to None.
        """

        super().__init__(name)
        self.bounds_merge_op = bounds_merge_op
        self.payload_merge_op = payload_merge_op
        self.interval_merge_op = interval_merge_op

        self.predicate = predicate
        self.epsilon = epsilon
        self.axis = axis

        self.publishable_coalesced_intrvls = []
        self.pending_coalesced_intrvls = []
        self.done = False

    def call(self, instream):
        self.instream = instream

    def execute(self):
        while True:
            if len(self.publishable_coalesced_intrvls) > 0 and \
                (len(self.pending_coalesced_intrvls) == 0 or \
                    self.publishable_coalesced_intrvls[0] < self.pending_coalesced_intrvls[0]):
                # the above condition ensures output t1 increases monotoically. 
                self.publish(self.publishable_coalesced_intrvls.pop(0))
                return True
            elif self.done:
                return False

            intrvl = self.instream.get()
            # logger.debug(f"got 1 input: {intrvl}")
            if intrvl is None:
                self.publishable_coalesced_intrvls = sorted(self.publishable_coalesced_intrvls +  self.pending_coalesced_intrvls)
                self.pending_coalesced_intrvls.clear()
                self.done = True
                continue

            # logger.debug(f"publishable: {self.publishable_coalesced_intrvls}")
            # logger.debug(f"pending: {self.pending_coalesced_intrvls}")

            bounds_merge_op = self.bounds_merge_op
            payload_merge_op = self.payload_merge_op
            interval_merge_op = self.interval_merge_op

            predicate = self.predicate
            epsilon = self.epsilon
            axis = self.axis

            new_coalesced_intrvls = self.publishable_coalesced_intrvls
            current_intrvls = self.pending_coalesced_intrvls
            new_current_intrvls = []

            for cur in current_intrvls:
                if Bounds.cast({
			        axis[0] : 't1',
			        axis[1] : 't2'
		        })(or_pred(overlaps(),
                    before(max_dist=epsilon)))(cur, intrvl):
                        #adds overlapping intervals to new_current_intrvls
                        new_current_intrvls.append(cur)            
                else:
                    #adds all non-overlapping intervals to new_coalesced_intrvls
                    logger.debug(f"Adding to publishable: {cur}")
                    new_coalesced_intrvls.append(cur)

            current_intrvls = new_current_intrvls
            matched_intrvl = None
            loc = len(current_intrvls) - 1

            #if current_intrvls is empty, we need to start constructing a new set of coalesced intervals
            if len(current_intrvls) == 0:
                current_intrvls.append(intrvl.copy())
                self.publishable_coalesced_intrvls = sorted(new_coalesced_intrvls)
                self.pending_coalesced_intrvls = sorted(current_intrvls)
                continue
            
            if predicate is None:
                matched_intrvl = current_intrvls[-1]
            else:
                for index, cur in enumerate(current_intrvls):
                    if predicate(cur, intrvl):
                        matched_intrvl = cur
                        loc = index

            #if no matching interval is found, this implies that intrvl should be the start of a new coalescing interval
            if matched_intrvl is None:
                current_intrvls.append(intrvl)
            else:
                # merge matched intrvl with the new intrvl
                if interval_merge_op is not None:
                    current_intrvls[loc] = interval_merge_op(matched_intrvl, intrvl)
                else:
                    current_intrvls[loc] = Interval(
                            bounds_merge_op(matched_intrvl['bounds'],
                                            intrvl['bounds']),
                            payload_merge_op(matched_intrvl['payload'],
                                            intrvl['payload'])
                        )

            self.publishable_coalesced_intrvls = sorted(new_coalesced_intrvls)
            self.pending_coalesced_intrvls = sorted(current_intrvls)


class CoalesceByLast(Graph):
    """Similar to ``Coalesce``, but ``predicate`` is applied differretly:
    For each pending coalesced interval,
    this op keeps track of the last input interval that was merged.
    When deciding whether a new input interval should be merged, ``predicate``
    is applied on the tracked "last" interval and the incoming interval,
    rather than the whole pending interval and the incoming interval
    (which ``Coalesce`` does).

    This op is useful for tracking an object that's slowly moving in the scene.
    If using ``Coalesce``, IoU test will at some point fail because the coalesced
    interval grows large enough as a result of spanning. By contrast, ``Coalesce``
    is suitable for mostly static objects.
    """
    
    def __init__(self, 
                bounds_merge_op=Bounds3D.span,
                payload_merge_op=lambda p1, p2: p1,
                predicate=None,
                epsilon=0,
                axis=('t1', 't2'),
                interval_merge_op=None):

        super().__init__()

        self.bounds_merge_op = bounds_merge_op
        self.payload_merge_op = payload_merge_op
        self.predicate = predicate
        self.predicate = predicate
        self.epsilon = epsilon
        self.axis = axis
        self.interval_merge_op = interval_merge_op

        # a secret key to track the last interval, should be removed when finish
        self.key_last = "coalesce_last_" + str(uuid.uuid4())

    def call(self, instream):

        def predicate_1(pending_i, new_i):
            last_i = pending_i.payload.get(self.key_last, pending_i)
            return self.predicate(last_i, new_i)

        def interval_merge_op_1(pending_i, new_i):
            if self.interval_merge_op:
                merged_i = self.interval_merge_op(pending_i, new_i)
            else:
                merged_i = Interval(
                    self.bounds_merge_op(pending_i.bounds, new_i.bounds),
                    self.payload_merge_op(pending_i.payload, new_i.payload)
                )
            # crucial: track the last merged interval
            merged_i.payload[self.key_last] = new_i
            return merged_i

        def remove_key_last(i):
            try:
                del i.payload[self.key_last]
            except KeyError:
                pass
            return i

        coalesced_stream = Coalesce(
            predicate=predicate_1,
            epsilon=self.epsilon,
            axis=self.axis,
            interval_merge_op=interval_merge_op_1
        )(instream)
        removed_key_stream = Map(remove_key_last)(coalesced_stream)
        return removed_key_stream


class Fold(Op):

    def __init__(
        self,
        update_fn: typing.Callable[[typing.Any, Interval], typing.Any],
        init_state: typing.Any = None,
        finalize_fn: typing.Callable[[typing.Any], Interval] = lambda x: x,
        name = None
    ):
        """Folds a finite interval stream into a single output interval.
        Similar to Rekall's ``IntervalSet.fold``, DBMS's aggregate, and MapReduce's reduce (though not necessarily commutative).
        It keeps an internal state and repeatedly updates it using incoming intervals,
        until the input stream is finished. The state can be any type, not necessarily an Interval.
        This Op should not be used on an infinite stream.

        Args:
            update_fn (typing.Callable[[typing.Any, Interval], typing.Any]): Update the internal state using a new interval. Return the new state.
            init_state (typing.Any, optional): Init state. If None, will use the first interval as state. Defaults to None.
            finalize_fn (typing.Callable[[typing.Any], Interval], optional): Convert the final state into an interval. Defaults to the identify function.
            name ([type], optional): [description]. Defaults to None.
        """
        super().__init__(name)
        self.update_fn = update_fn
        self.init_state = init_state
        self.finalize_fn = finalize_fn

        self.done = False

    def call(self, instream):
        self.instream = instream

    def execute(self) -> bool:
        if self.done:
            return False

        state = self.init_state
        while True:
            intrvl = self.instream.get()
            if intrvl is None:
                break   # Done
            elif state is None:   # no user provided state
                state = intrvl
                continue
            else:
                state = self.update_fn(state, intrvl)

        self.done = True        
        result = self.finalize_fn(state)
        self.publish(result)
        return True

class JoinWithTimeWindow(Op):
    """
    Join two ``IntervalStream`` by finding pairs that satisfy the join predicate. 
    Instead of checking the full Cartessian product, it only examines pairs of intervals
    whose ``t1`` are within ``window`` of each other. 

    For the output to maintain the temporal assumption, the following conditions must hold:
    1. The two input streams meet the temporal assumption
    2. The join result's temporal bound is "contained" within the temporal span of the joined pair.
    For example, joining (t1=10, t2=30) and (t1=15, t2=45) may produce (t1=10, t2=40). 
    Producing (t1=5, t2=60) violates this assumption, and the temporal order of the output
    cannot be guaranteed.

    """

    def __init__(
        self,
        predicate: typing.Callable[[Interval, Interval], bool], 
        merge_op: typing.Callable[[Interval, Interval], Interval],
        window = 900,
        name = None
    ):
        """[summary]

        Args:
            predicate (typing.Callable[[Interval, Interval], bool]): The predicate function that will be checked on pairs within ``window`` of each other
            merge_op (typing.Callable[[Interval, Interval], Interval]): Produce an output ``Interval`` based on a joined pair of ``Interval``
            window (int, optional): Windows on ``t1``. Defaults to 900.
            name ([type], optional): [description]. Defaults to None.
        """
        super().__init__(name)
        self.predicate = predicate
        self.merge_op = merge_op
        self.window = window

    def call(self, left_stream, right_stream):
        self.left_stream = left_stream
        self.right_stream = right_stream
    
        self.left_done = self.right_done = False
        self.left_buf = []
        self.right_buf = []
        self.result_buf = []    # keep sorted on t1

    def execute(self):
        # When we get a new input `iL` from left, we:
        # 1. purge right_buf of intervals that are too old before `iL`'s window
        # 2. join `iL` with right_buf considering `predicate` and window. Joined results are added to `result_buf`
        # 3. append `iL` to left_buf
        # Mirror operations on the right side.

        # a buffered result can be released only when the heads (earliest) of left_buf and right_buf are 
        # outside its window. Otherwise, a new incoming interval can generate a join result that
        # should be output first

        while not (self.left_done and self.right_done):
            if (len(self.result_buf) > 0
                and self.result_buf[0]['t1'] + self.window < self.left_buf[0]['t1']
                and self.result_buf[0]['t1'] + self.window < self.right_buf[0]['t1']):
                # we can release a result
                # logger.debug(f"releasing {self.result_buf[0].payload['msg']}")
                self.publish(self.result_buf.pop(0))
                return True
                
            if not self.left_done:
                # get a new interval from left
                iL = self.left_stream.get()
                if iL is None:
                    self.left_done = True
                else:
                    # logger.debug(f"get left {iL.payload['msg']}")
                    # add to buffer
                    self.left_buf.append(iL)
                    # purge too-old intervals from right buffer
                    self.right_buf = [iR for iR in self.right_buf if iR['t1'] >= iL['t1'] - self.window ]
                    # probe the right buf and join
                    hits = [
                        self.merge_op(iL, iR) for iR in self.right_buf 
                        if iR['t1'] <= iL['t1'] + self.window and self.predicate(iL, iR)
                    ]

                    self.result_buf.extend(hits)
                    self.result_buf.sort(key=lambda i: (i['t1'], i['t2']))
            
            # mirror
            if not self.right_done:
                iR = self.right_stream.get()
                if iR is None:
                    self.right_done = True
                else:
                    # logger.debug(f"get right {iR.payload['msg']}")
                    self.right_buf.append(iR)
                    self.left_buf = [iL for iL in self.left_buf if iL['t1'] >= iR['t1'] - self.window]
                    hits = [
                        self.merge_op(iL, iR) for iL in self.left_buf
                        if iL['t1'] <= iR['t1'] + self.window and self.predicate(iL, iR)
                    ]

                    self.result_buf.extend(hits)
                    self.result_buf.sort(key=lambda i: (i['t1'], i['t2']))

        if len(self.result_buf) > 0:
            self.publish(self.result_buf.pop(0))
            return True
        else:
            return False


# alias
Join = JoinWithTimeWindow