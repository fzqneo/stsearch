from queue import Queue 

from stsearch.interval import Interval

class IntervalStream(object):
    """An ``IntervalStream`` is a logical representation of the output from an ``Op``.
    Because we use a push-based mechanism with possible forks (i.e., the output of an Op
    is consumed by several other Op), the ``IntervalStream`` class doesn't buffer or queue
    the data itself, but instead delegate it to the ``IntervalStreamSubscriber`` class.
    This class provides interfaces for the Op to publish and subscribe (i.e., to create
    ``IntervalStreamSubscriber`` instances) results.
    """
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.subscribers = list()
        self.closed = False

    def publish(self, i):
        assert not self.closed, "Trying to publish to a closed IntervalStream"
        for sub in self.subscribers:
            sub.put(i)
        if i is None:
            self.closed = True

    def subscribe(self, *args, **kwargs):
        sub = IntervalStreamSubscriber(self, *args, **kwargs)
        self.subscribers.append(sub)
        return sub

    def start_thread_recursive(self):
        self.parent.start_thread_recursive()


class IntervalStreamSubscriber(object):
    def __init__(self, publisher, maxsize=512):
        super().__init__()
        assert isinstance(publisher, IntervalStream)
        self.q = Queue(maxsize=maxsize)
        self.publisher = publisher
        self.parent = publisher.parent

    def put(self, i):
        self.q.put(i)

    def get(self):
        return self.q.get()
        
    def __iter__(self):
        return self

    def __next__(self):
        rv = self.get()
        if rv is None:
            raise StopIteration
        else:
            return rv