import rekall

class Interval(rekall.Interval):
    def __init__(self, bounds, payload=None):
        # we force payload to be a dict
        assert payload is None or isinstance(payload, dict)
        super().__init__(bounds, payload or dict())
    
    def copy(self):
        return Interval(self.bounds.copy(), self.payload)

    def __str__(self):
        return "< Interval {} payload: {} >".format(
            self.bounds,
            [ str(k)+":"+type(v).__name__ for k, v in self.payload.items() ]
        )