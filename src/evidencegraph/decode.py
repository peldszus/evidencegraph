# -*- coding: utf-8 -*-

"""
@author: Andreas Peldszus
"""
from __future__ import absolute_import

import networkx as nx
import copy
from operator import itemgetter

from .depparse.graph import Digraph as DepDigraph
from collections import defaultdict
from .argtree import ArgTree


def multidigraph_to_digraph(g, field="weight", func=max):
    """
    Returns a DiGraph from a MultiDiGraph. If there are multiple edges
    from one node to the other, the edges is chosen which has the
    minimal or maximal value in some data field.

    >>> g = nx.MultiDiGraph()
    >>> g.add_edge(1, 2, weight=0.2, count=1000)
    >>> g.add_edge(1, 2, weight=0.8, count=0)
    >>> g.add_edge(2, 1, weight=0.4, count=1000)
    >>> g.add_edge(2, 1, weight=0.6, count=0)
    >>> d = multidigraph_to_digraph(g)
    >>> isinstance(d, nx.DiGraph)
    True
    >>> d.edges(data=True) == [(1, 2, {'count': 0, 'weight': 0.8}), (2, 1, {'count': 0, 'weight': 0.6})]
    True
    """
    f = nx.DiGraph()
    f.graph = g.graph
    for n in g.nodes():
        for m in g.succ[n].keys():
            # pick best edge
            ds = g.succ[n][m].values()
            d = func(ds, key=itemgetter(field))
            f.add_edge(n, m, **d)
    return f


def nxdigraph_to_depdigraph(g, field="weight"):
    """
    Returns a depparse.Digraph from an nx.DiGraph.
    `field` is the name of the data field to get the score from.

    >>> g = nx.DiGraph()
    >>> g.add_edge(1, 2, weight=0.3)
    >>> g.add_edge(2, 3, weight=0.8)
    >>> g.add_edge(3, 1, weight=0.1)
    >>> g.add_edge(2, 1, weight=0.5)
    >>> d = nxdigraph_to_depdigraph(g)
    >>> list(d.iteredges())
    [(1, 2), (2, 1), (2, 3), (3, 1), ('root', 1), ('root', 2), ('root', 3)]
    """
    succs = defaultdict(list)
    weights = {}
    for s, t, d in g.edges(data=True):
        w = d[field]
        succs[s].append(t)
        succs[t]
        weights[(s, t)] = w
    succs["root"] = list(succs.keys())
    weights.update({("root", n): 0 for n in succs.keys()})
    return DepDigraph(succs, get_score=lambda s, t: weights[(s, t)])


def find_mst(weg, from_root=False, field="weight"):
    """
    Returns the ArgTree that is the minimum spanning tree
    for the given nx.MultiDiGraph (such as e.g. an WeightedEvidenceGraph).

    >>> g = nx.MultiDiGraph()
    >>> g.add_edge(1, 2, weight=0.3)
    >>> g.add_edge(2, 3, weight=0.8)
    >>> g.add_edge(2, 3, weight=0.7)
    >>> g.add_edge(3, 1, weight=0.1)
    >>> g.add_edge(2, 1, weight=0.5)
    >>> mst = find_mst(g)
    >>> mst.edges(data=True)
    [(1, 2, {'weight': 0.3}), (2, 3, {'weight': 0.8})]
    """
    # make digraph of multidigraph
    deg = multidigraph_to_digraph(weg, field=field)

    # reverse graph if needed
    g = deg if from_root else deg.reverse()

    # convert to depparse digraph
    dd = nxdigraph_to_depdigraph(g, field=field)

    # find mst
    m = dd.mst()

    # convert depparse mst to ArgTree
    out = ArgTree()
    out.graph = copy.deepcopy(g.graph)
    out.add_nodes_from([copy.deepcopy((n, d)) for n, d in g.nodes(data=True)])
    for s, t in m.iteredges():
        if s == "root":
            # don't add the root link
            continue
        out.add_edge(s, t, g.succ[s][t])

    return out if from_root else out.reverse(copy=False)
