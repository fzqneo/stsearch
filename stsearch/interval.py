import rekall

class Interval(rekall.Interval):
    def __init__(self, bounds, payload=None):
        # we force payload to be a dict
        assert payload is None or isinstance(payload, dict)
        super().__init__(bounds, payload or dict())
    
