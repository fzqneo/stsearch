from queue import Queue 

from stsearch.interval import Interval

class IntervalStream(object):
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
    def __init__(self, publisher, maxsize=100):
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