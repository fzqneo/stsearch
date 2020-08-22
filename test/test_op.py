import logging
import unittest

from logzero import logger

from rekall import Bounds3D
from rekall.predicates import iou_at_least

from stsearch.interval import *
from stsearch.op import *

logger.setLevel(logging.INFO)

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


class OpTestCase(unittest.TestCase):
    def assertIntervalsEq(self, intrvl1, intrvl2, bounds_only=False):
        self.assertEqual(intrvl1['bounds'].data, intrvl2['bounds'].data)
        if not bounds_only:
            # only works for single-level dict with primitive value types
            self.assertDictEqual(intrvl1['payload'], intrvl2['payload'])

    def assertIntervalListEq(self, ilist1, ilist2, bounds_only=False):
        self.assertEqual(len(ilist1), len(ilist2))
        for i, j in zip(ilist1, ilist2):
            self.assertIntervalsEq(i, j, bounds_only=bounds_only)

class TestBasicOp(OpTestCase):

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


class TestSetOp(OpTestCase):

    def test_set_map(self):
        map_fn = lambda intrvl: Interval(
            intrvl.bounds.copy(),
            {'msg': intrvl.payload['msg'] + " bravo!"}
        )

        output = SetMap(map_fn)(FromIterable(create_intrvl_list_hello())())
        results = run_to_finish(output)
        self.assertListEqual(
            [intrvl.payload['msg'] for intrvl in results],
            ["hello bravo!", "world bravo!", "hello world bravo!"]
        )


class TestCoalesce(OpTestCase):

    intrvl_list_no_overlap = [ 
        Interval(Bounds3D(0, 10), {'msg': 'hello'}),
        Interval(Bounds3D(20, 30), {'msg': 'world'}),
        Interval(Bounds3D(40, 50), {'msg': 'hello world'}),
        Interval(Bounds3D(100, 200), {'msg': 'hello galaxy'}),
    ]

    intrvl_list_all_overlap  = [ 
        Interval(Bounds3D(0, 20), {'msg': 'hello'}),
        Interval(Bounds3D(15, 50), {'msg': 'world'}),
        Interval(Bounds3D(49, 60), {'msg': 'hello world'}),
    ]

    intrvl_list_some_overlap  = [ 
        Interval(Bounds3D(0, 20), {'msg': 'first'}),
        Interval(Bounds3D(15, 50), {'msg': 'second overlaps with first'}),
        Interval(Bounds3D(53, 60), {'msg': 'third is only 3 apart from second'}),
        Interval(Bounds3D(1000, 2000), {'msg': 'fourth is far part from third'}),
        Interval(Bounds3D(1999, 2999), {'msg': 'five overlaps with fourth'}),
    ]

    intrvl_list_iou = [
        Interval(Bounds3D(0, 10, 0., .5, 0., .5), {'msg': 'first person starts. frame 0~10'}),
        Interval(Bounds3D(10, 20, 0.01, .51, 0.01, .51), {'msg': 'first person stays at frame 10~20'}),
        Interval(Bounds3D(20, 30, 0.02, .52, 0.02, .52), {'msg': 'first person stays at frame 20~30. Ends'}),
     
        Interval(Bounds3D(10, 20, 0.5, 1., .5, 1.), {'msg': 'second person starts. frame 10~20'}),
        Interval(Bounds3D(20, 25, 0.49, .99, .49, .99), {'msg': 'second person ends. frame 20~25'}),

        Interval(Bounds3D(100, 110, 0., .8, .0, .8), {'msg': 'third person starts large. frame 100~110'}),
        Interval(Bounds3D(110, 120, 0., .79, .0, .79), {'msg': 'third person shrinks. frame 110~120'}),
    ]


    intrvl_list_iou_early_late = [
        # second person starts later but ends earlier than first person. Expect correct output order by t1.
        Interval(Bounds3D(0, 5, 0., .5, 0., .5), {'msg': 'first person starts at frame 0'}),
        Interval(Bounds3D(5, 10, .01, .51, .01, .51), {'msg': 'first person continues'}),
        Interval(Bounds3D(10, 11, 0.02, .52, 0.02, .52), {'msg': 'first person ends at frame 10'}),
     
        Interval(Bounds3D(3, 4, 0.5, 1., .5, 1.), {'msg': 'second person starts later but ends earlier than first person'}),
        Interval(Bounds3D(4, 8, 0.49, .99, .49, .99), {'msg': 'second person ends'}),

        Interval(Bounds3D(100, 110, 0., .8, .0, .8), {'msg': 'third person starts large. frame 100~110'}),
        Interval(Bounds3D(110, 120, 0., .79, .0, .79), {'msg': 'third person shrinks. frame 110~120'}),
    ]

    
    intrvl_list_iou_early_late_epsilon = [
        Interval(Bounds3D(0, 1, 0., .5, 0., .5), {'msg': 'first person starts at frame 0'}),
        Interval(Bounds3D(5, 6, .01, .51, .01, .51), {'msg': 'first person continues'}),
        Interval(Bounds3D(10, 11, 0.02, .52, 0.02, .52), {'msg': 'first person ends at frame 10'}),
     
        Interval(Bounds3D(3, 4, 0.5, 1., .5, 1.), {'msg': 'second person starts later but ends earlier than first person'}),
        Interval(Bounds3D(7, 8, 0.49, .99, .49, .99), {'msg': 'second person ends'}),

        Interval(Bounds3D(100, 110, 0., .8, .0, .8), {'msg': 'third person starts large. frame 100~110'}),
        Interval(Bounds3D(115, 120, 0., .79, .0, .79), {'msg': 'third person shrinks. frame 110~120'}),
    ]


    def test_set_coalesce_no_overlap(self):
        output = SetCoalesce(
            axis=('t1', 't2'), bounds_merge_op=Bounds3D.span)(
                FromIterable(self.intrvl_list_no_overlap)())
        results = run_to_finish(output)
        self.assertIntervalListEq(results, self.intrvl_list_no_overlap)

    def test_set_coalesce_all_overlap(self):
        output = SetCoalesce(
            axis=('t1', 't2'), bounds_merge_op=Bounds3D.span)(
                FromIterable(self.intrvl_list_all_overlap)())
        results = run_to_finish(output)
        self.assertIntervalListEq(
            results,
            [Interval(Bounds3D(0, 60), {'msg': 'hello'})]
        )

    def test_set_coalesce_all_overlap_payload_merge(self):
        def payload_merge_msg(p1, p2):
            return {'msg': p1['msg'] + '|' + p2['msg']}

        output = SetCoalesce(
            axis=('t1', 't2'), bounds_merge_op=Bounds3D.span, payload_merge_op=payload_merge_msg)(
                FromIterable(self.intrvl_list_all_overlap)())
        results = run_to_finish(output)
        self.assertIntervalListEq(
            results,
            [Interval(Bounds3D(0, 60), {'msg': 'hello|world|hello world'})]
        )

    def test_set_coalesce_some_overlap(self):
        output = SetCoalesce(
            axis=('t1', 't2'), bounds_merge_op=Bounds3D.span)(
                FromIterable(self.intrvl_list_some_overlap)())
        results = run_to_finish(output)
        L = self.intrvl_list_some_overlap
        self.assertIntervalListEq(
            results,
            [
                Interval(Bounds3D(0, 50), L[0].payload),    # 0 and 1 merge
                Interval(Bounds3D(53, 60), L[2].payload ),  # 2 unchanged
                Interval(Bounds3D(1000, 2999), L[3].payload), # 3 and 4 merge
            ]
        )

    def test_set_coalesce_some_overlap_epsilon(self):
        output = SetCoalesce(
            axis=('t1', 't2'), bounds_merge_op=Bounds3D.span, epsilon=5)(
                FromIterable(self.intrvl_list_some_overlap)())
        results = run_to_finish(output)
        L = self.intrvl_list_some_overlap
        self.assertIntervalListEq(
            results,
            [
                Interval(Bounds3D(0, 60), L[0].payload),    # 0, 1, 2 merge
                Interval(Bounds3D(1000, 2999), L[3].payload), # 3 and 4 merge
            ]
        )


    def test_set_coalesce_some_overlap_epsilon_2(self):
        output = SetCoalesce(
            axis=('t1', 't2'), bounds_merge_op=Bounds3D.span, epsilon=999)(
                FromIterable(self.intrvl_list_some_overlap)())
        results = run_to_finish(output)
        L = self.intrvl_list_some_overlap
        self.assertIntervalListEq(
            results,
            [
                Interval(Bounds3D(0, 2999), L[0].payload),    # all should merge
            ]
        )

    def test_set_coalesce_iou(self):

        input_list = self.intrvl_list_iou

        target = [
            Interval(Bounds3D(0, 30, 0., .52, 0., .52), {'msg': 'first person starts.'}),
            Interval(Bounds3D(10, 25, 0.49, 1., .49, 1.), {'msg': 'second person starts.'}),
            Interval(Bounds3D(100, 120, 0., .8, .0, .8), {'msg': 'third person starts large.'}),
        ]

        output = SetCoalesce(
            axis=('t1', 't2'), bounds_merge_op=Bounds3D.span, predicate=iou_at_least(0.8))(
                FromIterable(input_list)())
        results = run_to_finish(output)
        # print(results)
        self.assertIntervalListEq(results, target, bounds_only=True)


    def test_coalesce_iou(self):

        input_list = self.intrvl_list_iou

        target = [
            Interval(Bounds3D(0, 30, 0., .52, 0., .52), {'msg': 'first person starts.'}),
            Interval(Bounds3D(10, 25, 0.49, 1., .49, 1.), {'msg': 'second person starts.'}),
            Interval(Bounds3D(100, 120, 0., .8, .0, .8), {'msg': 'third person starts large.'}),
        ]

        output = Coalesce(
            axis=('t1', 't2'), bounds_merge_op=Bounds3D.span, predicate=iou_at_least(0.8))(
                FromIterable(input_list)())
        results = run_to_finish(output)
        # print(results)
        self.assertIntervalListEq(results, target, bounds_only=True)


    def test_set_coalesce_iou_early_late(self):

        input_list = self.intrvl_list_iou_early_late

        target = [
            Interval(Bounds3D(0, 11, 0., .52, 0., .52), {'msg': 'first person .'}),
            Interval(Bounds3D(3, 8, 0.49, 1., .49, 1.), {'msg': 'second person .'}),
            Interval(Bounds3D(100, 120, 0., .8, .0, .8), {'msg': 'third person starts .'}),
        ]

        output = SetCoalesce(
            axis=('t1', 't2'), bounds_merge_op=Bounds3D.span, predicate=iou_at_least(0.8))(
                FromIterable(input_list)())
        results = run_to_finish(output)
        # print(results)
        self.assertIntervalListEq(results, target, bounds_only=True)


    def test_coalesce_iou_early_late(self):

        input_list = self.intrvl_list_iou_early_late

        target = [
            Interval(Bounds3D(0, 11, 0., .52, 0., .52), {'msg': 'first person .'}),
            Interval(Bounds3D(3, 8, 0.49, 1., .49, 1.), {'msg': 'second person .'}),
            Interval(Bounds3D(100, 120, 0., .8, .0, .8), {'msg': 'third person starts .'}),
        ]

        output = Coalesce(
            axis=('t1', 't2'), bounds_merge_op=Bounds3D.span, predicate=iou_at_least(0.8))(
                FromIterable(input_list)())
        results = run_to_finish(output)
        # print(results)
        self.assertIntervalListEq(results, target, bounds_only=True)

    
    def test_set_coalesce_iou_early_late_epsilon(self):

        input_list = self.intrvl_list_iou_early_late_epsilon

        target = [
            Interval(Bounds3D(0, 11, 0., .52, 0., .52), {'msg': 'first person .'}),
            Interval(Bounds3D(3, 8, 0.49, 1., .49, 1.), {'msg': 'second person .'}),
            Interval(Bounds3D(100, 120, 0., .8, .0, .8), {'msg': 'third person starts .'}),
        ]

        output = SetCoalesce(
            axis=('t1', 't2'), bounds_merge_op=Bounds3D.span, predicate=iou_at_least(0.8), epsilon=6)(
                FromIterable(input_list)())
        results = run_to_finish(output)
        # print(results)
        self.assertIntervalListEq(results, target, bounds_only=True)


    def test_coalesce_iou_early_late_epsilon(self):

        input_list = self.intrvl_list_iou_early_late_epsilon

        target = [
            Interval(Bounds3D(0, 11, 0., .52, 0., .52), {'msg': 'first person .'}),
            Interval(Bounds3D(3, 8, 0.49, 1., .49, 1.), {'msg': 'second person .'}),
            Interval(Bounds3D(100, 120, 0., .8, .0, .8), {'msg': 'third person starts .'}),
        ]

        output = Coalesce(
            axis=('t1', 't2'), bounds_merge_op=Bounds3D.span, predicate=iou_at_least(0.8), epsilon=6)(
                FromIterable(input_list)())
        results = run_to_finish(output)
        # print(results)
        self.assertIntervalListEq(results, target, bounds_only=True)


        
class TestFold(OpTestCase):

    intrvl_list_number = [
        Interval(Bounds3D(0, 10), {'number': 2}),
        Interval(Bounds3D(20, 30), {'number': 10}),
        Interval(Bounds3D(40, 50), {'number': 20}),
        Interval(Bounds3D(100, 200), {'number': 100}),
    ]

    def test_fold_number_max(self):
        input_list = self.intrvl_list_number

        target = [
            Interval(Bounds3D(0, 0), {'number': 100}),
        ]

        output = Fold(
            update_fn=lambda state, intrvl: max(state, intrvl.payload['number']),
            init_state=0.,
            finalize_fn=lambda state: Interval(Bounds3D(0, 0), {'number': state}),
        )(FromIterable(input_list)())

        results = run_to_finish(output)
        self.assertIntervalListEq(results, target)


    def test_fold_number_min(self):
        input_list = self.intrvl_list_number

        target = [
            Interval(Bounds3D(0, 0), {'number': 2}),
        ]

        output = Fold(
            update_fn=lambda state, intrvl: min(state, intrvl.payload['number']),
            init_state=float('inf'),
            finalize_fn=lambda state: Interval(Bounds3D(0, 0), {'number': state}),
        )(FromIterable(input_list)())

        results = run_to_finish(output)
        self.assertIntervalListEq(results, target)


    def test_fold_number_avg(self):
        input_list = self.intrvl_list_number

        target = [
            Interval(Bounds3D(0, 0), {'number': 33}),
        ]

        # state = (count, sum)
        output = Fold(
            update_fn=lambda state, intrvl: (state[0]+1, state[1]+intrvl.payload['number']),
            init_state=(0, 0.),
            finalize_fn=lambda state: Interval(Bounds3D(0, 0), {'number': state[1]/state[0]}),
        )(FromIterable(input_list)())

        results = run_to_finish(output)
        self.assertIntervalListEq(results, target)

