# The MIT License (MIT)
#
# Copyright (c) 2014 Leif Johnson
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import collections
import logging
import unittest

from .graph import Digraph


class CycleTest(unittest.TestCase):
    def _test_cycle(self, succs):
        logging.info(succs)
        graph = Digraph(succs)
        logging.info(graph.dot('g'))
        cycle = graph.find_cycle()
        logging.info(cycle.dot('cycle'))
        return cycle

    def test_one(self):
        self.assertEqual(1, self._test_cycle({1: [1]}).num_edges())

    def test_two(self):
        self.assertEqual(2, self._test_cycle({1: [2], 2: [1]}).num_edges())

    def test_three(self):
        self.assertEqual(
            3, self._test_cycle({1: [2], 2: [3], 3: [1]}).num_edges())

    def test_three_two(self):
        self.assertEqual(
            2, self._test_cycle({1: [2], 2: [1, 3], 3: [1]}).num_edges())


class MstTest(unittest.TestCase):
    def _test_chuliuedmonds(self, scores, cycles_and_edges):
        logging.info('**********')

        succs = collections.defaultdict(list)
        for s, t in scores:
            succs[s].append(t)
            succs[t]

        num_nodes = len(succs)
        num_edges = len(scores)

        graph = Digraph(succs,
                        lambda s, t: scores[s, t][0],
                        lambda s, t: scores[s, t][1])
        logging.info(graph.dot('g'))
        self.assertEqual(num_nodes, len(graph.successors))
        self.assertEqual(num_edges, len(tuple(graph.iteredges())))

        self.__test_chuliuedmonds(graph, num_nodes, num_edges, cycles_and_edges)

    def __test_chuliuedmonds(self, graph, num_nodes, num_edges, cycles_and_edges):
        greedy = graph.greedy()
        logging.info(greedy.dot('greedy'))
        self.assertEqual(num_nodes, len(greedy.successors))
        self.assertEqual(num_nodes - 1, len(tuple(greedy.iteredges())))

        cycle = greedy.find_cycle()
        if not cycles_and_edges:
            self.assert_(cycle is None)
            return greedy

        cycle_length, num_compact_edges = cycles_and_edges.pop(0)

        logging.info(cycle.dot('cycle'))
        self.assertEqual(cycle_length, len(cycle.successors))
        self.assertEqual(cycle_length, len(tuple(cycle.iteredges())))

        new_id, old_edges, compact = graph.contract(cycle)
        logging.info(compact.dot(new_id))
        self.assertEqual(num_nodes - cycle_length + 1, len(compact.successors))
        self.assertEqual(num_compact_edges, len(tuple(compact.iteredges())))

        again = self.__test_chuliuedmonds(compact,
                                          num_nodes - cycle_length + 1,
                                          num_compact_edges,
                                          cycles_and_edges)
        logging.info(again.dot('again'))
        self.assert_(again.find_cycle() is None)
        self.assertEqual(num_nodes - cycle_length + 1, len(again.successors))
        self.assertEqual(num_nodes - cycle_length, len(tuple(again.iteredges())))

        merged = graph.merge(again, new_id, old_edges, cycle)
        logging.info(merged.dot('merged'))
        self.assert_(merged.find_cycle() is None)
        self.assertEqual(num_nodes, len(merged.successors))
        self.assertEqual(num_nodes - 1, len(tuple(merged.iteredges())))

        return merged

    def test_john_saw_mary(self):
        self._test_chuliuedmonds({
            ('ROOT_0', 'John_1'): (9, '_'),
            ('ROOT_0', 'saw_2'): (10, 'ROOT'),
            ('ROOT_0', 'Mary_3'): (9, '_'),
            ('John_1', 'saw_2'): (20, '_'),
            ('John_1', 'Mary_3'): (3, '_'),
            ('saw_2', 'John_1'): (30, 'SBJ'),
            ('saw_2', 'Mary_3'): (30, 'OBJ'),
            ('Mary_3', 'John_1'): (11, '_'),
            ('Mary_3', 'saw_2'): (0, '_'),
            }, [(2, 4)])

    def test_the_boy_hit_the_ball(self):
        self._test_chuliuedmonds({
            ('ROOT_0', 'The_1'): (0, '_'),
            ('ROOT_0', 'boy_2'): (9, '_'),
            ('ROOT_0', 'hit_3'): (10, 'ROOT'),
            ('ROOT_0', 'the_4'): (0, '_'),
            ('ROOT_0', 'ball_5'): (9, '_'),

            ('The_1', 'boy_2'): (0, '_'),
            ('The_1', 'hit_3'): (0, '_'),
            ('The_1', 'the_4'): (0, '_'),
            ('The_1', 'ball_5'): (0, '_'),

            ('boy_2', 'The_1'): (5, 'DET'),
            ('boy_2', 'hit_3'): (20, '_'),
            ('boy_2', 'the_4'): (0, '_'),
            ('boy_2', 'ball_5'): (3, '_'),

            ('hit_3', 'The_1'): (0, '_'),
            ('hit_3', 'boy_2'): (30, 'SBJ'),
            ('hit_3', 'the_4'): (0, '_'),
            ('hit_3', 'ball_5'): (30, 'OBJ'),

            ('the_4', 'The_1'): (0, '_'),
            ('the_4', 'boy_2'): (0, '_'),
            ('the_4', 'hit_3'): (0, '_'),
            ('the_4', 'ball_5'): (0, '_'),

            ('ball_5', 'The_1'): (0, '_'),
            ('ball_5', 'boy_2'): (11, '_'),
            ('ball_5', 'hit_3'): (0, '_'),
            ('ball_5', 'the_4'): (5, 'DET'),

            }, [(2, 16)])

    def test_the_boy_hit_the_ball_with_the_bat(self):
        self._test_chuliuedmonds({
            ('ROOT_0', 'The_1'): (0, '_'),
            ('ROOT_0', 'boy_2'): (9, '_'),
            ('ROOT_0', 'hit_3'): (10, 'ROOT'),
            ('ROOT_0', 'the_4'): (0, '_'),
            ('ROOT_0', 'ball_5'): (9, '_'),
            ('ROOT_0', 'with_6'): (0, '_'),
            ('ROOT_0', 'the_7'): (0, '_'),
            ('ROOT_0', 'bat_8'): (0, '_'),

            ('The_1', 'boy_2'): (0, '_'),
            ('The_1', 'hit_3'): (0, '_'),
            ('The_1', 'the_4'): (0, '_'),
            ('The_1', 'ball_5'): (0, '_'),
            ('The_1', 'with_6'): (0, '_'),
            ('The_1', 'the_7'): (0, '_'),
            ('The_1', 'bat_8'): (0, '_'),

            ('boy_2', 'The_1'): (5, 'DET'),
            ('boy_2', 'hit_3'): (20, '_'),
            ('boy_2', 'the_4'): (0, '_'),
            ('boy_2', 'ball_5'): (3, '_'),
            ('boy_2', 'with_6'): (0, '_'),
            ('boy_2', 'the_7'): (0, '_'),
            ('boy_2', 'bat_8'): (0, '_'),

            ('hit_3', 'The_1'): (0, '_'),
            ('hit_3', 'boy_2'): (30, 'SBJ'),
            ('hit_3', 'the_4'): (0, '_'),
            ('hit_3', 'ball_5'): (30, 'OBJ'),
            ('hit_3', 'with_6'): (30, 'VMOD'),
            ('hit_3', 'the_7'): (0, '_'),
            ('hit_3', 'bat_8'): (0, '_'),

            ('the_4', 'The_1'): (0, '_'),
            ('the_4', 'boy_2'): (0, '_'),
            ('the_4', 'hit_3'): (0, '_'),
            ('the_4', 'ball_5'): (0, '_'),
            ('the_4', 'with_6'): (0, '_'),
            ('the_4', 'the_7'): (0, '_'),
            ('the_4', 'bat_8'): (0, '_'),

            ('ball_5', 'The_1'): (0, '_'),
            ('ball_5', 'boy_2'): (11, '_'),
            ('ball_5', 'hit_3'): (0, '_'),
            ('ball_5', 'the_4'): (5, 'DET'),
            ('ball_5', 'with_6'): (20, 'NMOD'),
            ('ball_5', 'the_7'): (0, '_'),
            ('ball_5', 'bat_8'): (0, '_'),

            ('with_6', 'The_1'): (0, '_'),
            ('with_6', 'boy_2'): (0, '_'),
            ('with_6', 'hit_3'): (0, '_'),
            ('with_6', 'the_4'): (0, '_'),
            ('with_6', 'ball_5'): (15, 'PMOD'),
            ('with_6', 'the_7'): (0, '_'),
            ('with_6', 'bat_8'): (20, 'PMOD'),

            ('the_7', 'The_1'): (0, '_'),
            ('the_7', 'boy_2'): (0, '_'),
            ('the_7', 'hit_3'): (0, '_'),
            ('the_7', 'the_4'): (0, '_'),
            ('the_7', 'ball_5'): (0, '_'),
            ('the_7', 'with_6'): (0, '_'),
            ('the_7', 'bat_8'): (0, '_'),

            ('bat_8', 'The_1'): (0, '_'),
            ('bat_8', 'boy_2'): (0, '_'),
            ('bat_8', 'hit_3'): (0, '_'),
            ('bat_8', 'the_4'): (0, '_'),
            ('bat_8', 'ball_5'): (0, '_'),
            ('bat_8', 'with_6'): (35, 'NMOD'),
            ('bat_8', 'the_7'): (5, 'DET'),

            }, [(2, 49), (2, 36)])
