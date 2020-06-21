import unittest

from rekall import Bounds3D

from stsearch.interval import *
from stsearch.op import *

def create_intrvl_list_3():
    return [ 
        Interval(Bounds3D(0, 1), {'name': 'first'}),
        Interval(Bounds3D(1, 2), {'name': 'second'}),
        Interval(Bounds3D(2, 3), {'name': 'third'}),

    ]

def create_intrvl_list_100():
    return [ 
        Interval(Bounds3D(j, j+1), {'name': 'value-'+str(j)})
        for j in range(100)
    ]

def create_intrvl_list_hello():
    return [ 
        Interval(Bounds3D(0, 1), {'msg': 'hello'}),
        Interval(Bounds3D(1, 2), {'msg': 'world'}),
        Interval(Bounds3D(2, 3), {'msg': 'hello world'}),
    ]

def run(ostream):
    resultsub = ostream.subscribe()
    ostream.start_thread_recursive()
    return resultsub

def run_to_finish(ostream):
    return list(iter(run(ostream)))


class TestBasicOp(unittest.TestCase):

    def test_from_iterable(self):
        for create_fn in (create_intrvl_list_3, create_intrvl_list_100):
            output = FromIterable(create_fn())()
            result = run(output)
            self.assertEqual(len(list(iter(result))), len(create_fn()))

    def test_filter_t0(self):
        output = Filter(pred_fn=lambda intrvl: intrvl['t1'] % 3 == 0)(
            FromIterable(create_intrvl_list_100())()
        )
        result = run(output)
        self.assertEqual(len(list(iter(result))), 34)

        output = Filter(pred_fn=lambda intrvl: intrvl['t1'] % 20 == 0)(
            FromIterable(create_intrvl_list_100())()
        )
        result = run(output)
        self.assertEqual(len(list(iter(result))), 5)

    def test_filter_payload(self):
        output = Filter(
            pred_fn=lambda intrvl: intrvl.payload['msg'].startswith('hello'))(
            FromIterable(create_intrvl_list_hello())()
        )
        result_intrvls = list(iter(run(output)))
        print(result_intrvls)
        self.assertListEqual(
            [intrvl.payload['msg'] for intrvl in result_intrvls],
            ['hello', 'hello world']
        )

    def test_map_payload(self):
        map_fn = lambda intrvl: Interval(
            intrvl.bounds.copy(),
            {'msg': intrvl.payload['msg'] + " bravo!"}
        )

        output = Map(map_fn)(FromIterable(create_intrvl_list_hello())())
        results = run_to_finish(output)
        self.assertListEqual(
            [intrvl.payload['msg'] for intrvl in results],
            ["hello bravo!", "world bravo!", "hello world bravo!"]
        )

    def test_slice(self):
        output = Slice(step=2)(FromIterable(create_intrvl_list_100())())
        results = run_to_finish(output)
        self.assertEqual(len(results), 50)

        output = Slice(start=10, end=50, step=2)(FromIterable(create_intrvl_list_100())())
        results = run_to_finish(output)
        self.assertEqual(len(results), 20)
        self.assertEqual(results[0]['t1'], 10)
        self.assertEqual(results[-1]['t1'], 48)
