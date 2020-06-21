import collections
import threading

from logzero import logger

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
    
    def __init__(self, name=None):
        super().__init__()
        self._inputs = None
        self.output = None
        self.started = False

        if name is not None:
            self.name = name
        else:
            self.name = self.__class__.__name__ + '-' + str(Op._name_counter[self.__class__.__name__])
            Op._name_counter[self.__class__.__name__] += 1

    def call(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
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

    def execute(self):
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

    def publish(self, i):
        assert i is None or isinstance(i, Interval)
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
    def __init__(self, map_fn, name=None):
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
    def __init__(self, pred_fn, name=None):
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
    def __init__(self, iterable_of_intervals, name=None):
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