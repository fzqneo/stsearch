from queue import Queue 

from stsearch.interval import Interval

class IntervalStream(object):
    def __init__(self, parent, maxsize=-1):
        super().__init__()
        self.parent = parent
        self.children  = list()
        self.q = Queue(maxsize)
        self.closed = False
    
    def __iter__(self):
        return self

    def __next__(self):
        v = self.get()
        if v is not None:
            return v
        else:
            raise StopIteration

    def get(self):
        if self.closed:
            return None
        elif self.q.empty():
            if not self.parent.execute():
                self.closed = True
                return None
        return self.q.get()
        
    def put(self, i):
        self.q.put(i)
        