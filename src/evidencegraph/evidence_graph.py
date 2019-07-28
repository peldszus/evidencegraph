# -*- coding: utf-8 -*-

'''
@author: Andreas Peldszus
'''
from __future__ import print_function

import networkx as nx


class EvidenceGraph(nx.MultiDiGraph):
    '''
    A (potentially fully connected) MultiDiGraph, where every edge has a vector
    of scores. From this vector of scores, a weighted sum can be calculated to
    assign a single weight to every edge in a newly formed WeightedEvidenceGraph.
    '''

    def __init__(self, data=None, main_weight_id="weight", weight_ids=None,
                 **attr):
        '''
        Constructs a new EvidenceGraph.
        `main_weight_id` is the name of the final total weight.
        `weight_ids` is an iterable with the ids of the different weights.

        >>> eg = EvidenceGraph(weight_ids=['cc', 'ro', 'fu', 'at'])
        >>> eg.main_weight_id == "weight"
        True
        >>> eg.weight_ids == ['cc', 'ro', 'fu', 'at']
        True
        '''
        self.main_weight_id = main_weight_id
        self.weight_ids = weight_ids if weight_ids is not None else list()
        super(EvidenceGraph, self).__init__(data, **attr)

    def get_weighted_evidence_graph(self, weights=None):
        '''
        Returns an isomorph WeightedEvidenceGraph, where every edge has
        as a single score the weighted sum of the various weights assigned
        in the evidence graph.

        >>> eg = EvidenceGraph(weight_ids=['cc', 'ro', 'fu', 'at'])
        >>> eg.add_edge(1, 2, type="rel_a", cc=0.1, ro=0.9, fu=0.1, at=0.9)
        >>> eg.add_edge(1, 2, type="rel_b", cc=0.9, ro=0.1, fu=0.9, at=0.1)
        >>> weg = eg.get_weighted_evidence_graph()
        >>> weg.edges(data=True)
        [(1, 2, {'type': 'rel_a', 'weight': 0.5}),
         (1, 2, {'type': 'rel_b', 'weight': 0.5})]
        >>> weg = eg.get_weighted_evidence_graph(weights={'cc': 0.5, 'ro': 0.3, 'fu': 0.2, 'at': 0.0})
        >>> weg.edges(data=True)
        [(1, 2, {'type': 'rel_a', 'weight': 0.34}),
         (1, 2, {'type': 'rel_b', 'weight': 0.66})]
        '''
        # copy the graph structure
        g = WeightedEvidenceGraph(main_weight_id=self.main_weight_id)
        g.graph = self.graph
        g.add_nodes_from(self.nodes(data=True))
        # default to equally weighted weights
        if weights is None:
            weights = {x: 1.0 for x in self.weight_ids}
        # add the used weighting to the graph data
        g.graph['weighting'] = weights
        # calculate the weighted sum for all weights
        for s, t, k, d in self.edges(keys=True, data=True):
            # keep all the dict entries but the weight_ids
            new_d = {u: v for u, v in d.iteritems()
                     if u not in self.weight_ids}
            # calc weight
            new_d[self.main_weight_id] = self._normalized_weighted_sum(
                d, weights)
            g.add_edge(s, t, key=k, attr_dict=new_d)
        return g

    def _normalized_weighted_sum(self, d, weights):
        sum_of_weights = 0.0
        sum_of_weighted_weights = 0.0
        for weight_id, weight in weights.iteritems():
            if weight_id not in self.weight_ids or weight_id not in d:
                print(("Warning: '%s' is not a registered weight id. "
                       "skipping.") % weight_id)
                continue
            else:
                w = d[weight_id]
                ww = weight
                sum_of_weights += ww
                sum_of_weighted_weights += (w * ww)
        return sum_of_weighted_weights / sum_of_weights


class WeightedEvidenceGraph(nx.MultiDiGraph):

    def __init__(self, data=None, main_weight_id="weight", **attr):
        self.main_weight_id = main_weight_id
        super(WeightedEvidenceGraph, self).__init__(data, **attr)
