# -*- coding: utf-8 -*-

"""
@author: Andreas Peldszus
"""
from __future__ import absolute_import

from collections import deque
from itertools import permutations

import networkx as nx

from .arggraph import ArgGraph


class RelationSet(object):
    def __init__(self, preserving, inverting):
        """
        A set of argumentative relations.

        >>> s = RelationSet(['a', 'b', 'c', 'd'], ['x', 'y'])
        >>> s.function_central_claim
        'cc'
        >>> s.functions_preserving_role
        ['a', 'b', 'c', 'd']
        >>> s.functions_inverting_role
        ['x', 'y']
        >>> s.functions
        ['cc', 'a', 'b', 'c', 'd', 'x', 'y']
        >>> sorted(s.map_function_to_vector.items())
        [('a', 1), ('b', 2), ('c', 3), ('cc', 0), ('d', 4), ('x', 5), ('y', 6)]
        """
        self.function_central_claim = "cc"
        assert self.function_central_claim not in preserving
        assert self.function_central_claim not in inverting
        self.functions_preserving_role = preserving
        self.functions_inverting_role = inverting
        self.functions = (
            [self.function_central_claim]
            + self.functions_preserving_role
            + self.functions_inverting_role
        )
        self.map_function_to_vector = {
            f: i for i, f in enumerate(self.functions)
        }
        self.map_vector_to_function = {
            i: f for f, i in self.map_function_to_vector.items()
        }


SIMPLE_RELATION_SET = RelationSet(["sup"], ["att"])
"""The reduced set of relations, suitable for both EDU and ADU segmentation."""

FULL_RELATION_SET = RelationSet(
    ["support", "example", "join", "link"], ["rebut", "undercut"]
)
"""The full set of relations, suitable for EDU segmentation, as it contains the
   join relation."""

FULL_RELATION_SET_ADU = RelationSet(
    ["support", "example", "link"], ["rebut", "undercut"]
)
"""The full set of relations, suitable for ADU segmentation, without the join
   relation."""

RELATION_SETS = [SIMPLE_RELATION_SET, FULL_RELATION_SET, FULL_RELATION_SET_ADU]
"""List of all predefined relationsets."""

RELATION_SETS_BY_NAME = {
    "SIMPLE_RELATION_SET": SIMPLE_RELATION_SET,
    "FULL_RELATION_SET": FULL_RELATION_SET,
    "FULL_RELATION_SET_ADU": FULL_RELATION_SET_ADU,
}
"""Mapping of name to object for all predefined relationsets."""


class ArgTree(nx.DiGraph):

    ROLES = {"pro": "opp", "opp": "pro"}
    """Roles and their inverse role"""

    def __init__(
        self,
        from_arggraph=None,
        from_triples=None,
        text_id=None,
        relation_set=SIMPLE_RELATION_SET,
    ):
        """
        A simple dependency tree representing argumentation structures. The
        relation set is restricted to the simple destinction between
        supporting ('sup') and attacking ('att') edges.

        Note: This data structure is expected to be initialized with an edge
        set that conforms with the tree contraints. However, tree-ness is
        neither enforced nor tested.

        Args:
            from_arggraph (optional: ArgGraph): load from this ArgGraph object
            from_triples (optional: iterable): load from this list of edges

        Raises:
            Exception: when both `from_arggraph` and `from_triples` are
                provided.

        >>> ArgTree() is not None
        True
        >>> t = ArgTree(from_arggraph="sth", from_triples="sth")
        Traceback (most recent call last):
        [...]
        Exception: Cannot load argtree from arggraph and from triples at the same time.
        """
        super(ArgTree, self).__init__()
        self.relation_set = relation_set

        if from_arggraph and from_triples:
            raise Exception(
                (
                    "Cannot load argtree from arggraph and from "
                    "triples at the same time."
                )
            )
        elif from_arggraph:
            self.load_from_arggraph(from_arggraph)
        elif from_triples:
            self.load_from_triples(from_triples)

        if text_id:
            self.graph["text"] = text_id

    @staticmethod
    def edu_triples_to_adu_triples(triples):
        """
        We drop all 'join' edges and keep the 'restate' ones.
        Then edges are renumbered.

        >>> t = [(2, 1, 'join'), (3, 1, 'reb'), (4, 3, 'join'), (5, 3, 'link'), (6, 3, 'und'), (7, 1, 'restate')]
        >>> ArgTree.edu_triples_to_adu_triples(t)
        [(2, 1, 'reb'), (3, 2, 'link'), (4, 2, 'und'), (5, 1, 'restate')]

        """
        triples_without_join = [t for t in triples if t[2] != "join"]
        all_ids = sorted(
            set(t[0] for t in triples_without_join)
            | set(t[1] for t in triples_without_join)
        )
        mapping = {old: new for new, old in enumerate(all_ids, 1)}
        return [
            (mapping[src], mapping[trg], rel)
            for src, trg, rel in triples_without_join
        ]

    def load_from_arggraph(self, g, from_adus=True, long_names=False):
        """
        Loads the ArgTree from an ArgGraph.

        >>> from .arggraph import get_very_complex_arggraph
        >>> g = get_very_complex_arggraph()
        >>> t = ArgTree(relation_set=SIMPLE_RELATION_SET)
        >>> t.load_from_arggraph(g)
        >>> t.graph['text']
        'g1'
        >>> t.edges(data=True)
        [(2, 1, {'type': 'att'}), (3, 2, {'type': 'sup'}),
         (4, 2, {'type': 'att'}), (5, 1, {'type': 'sup'})]

        >>> g = get_very_complex_arggraph()
        >>> t = ArgTree(relation_set=FULL_RELATION_SET)
        >>> t.load_from_arggraph(g, from_adus=False, long_names=True)
        >>> t.graph['text']
        'g1'
        >>> t.edges(data=True)
        [(2, 1, {'type': 'join'}), (3, 1, {'type': 'rebut'}),
         (4, 3, {'type': 'join'}), (5, 3, {'type': 'link'}),
         (6, 3, {'type': 'undercut'}), (7, 1, {'type': 'restate'})]

        >>> t = ArgTree(relation_set=FULL_RELATION_SET)
        >>> t.load_from_arggraph(g, from_adus=True, long_names=False)
        >>> t.graph['text']
        'g1'
        >>> t.edges(data=True)
        [(2, 1, {'type': 'reb'}), (3, 2, {'type': 'link'}),
         (4, 2, {'type': 'und'}), (5, 1, {'type': 'restate'})]

        """
        assert isinstance(g, ArgGraph)
        # get dependencies according to segmentation
        r = g.get_edus_as_dependencies(include_cc=False, ids_to_numbers=True)
        if from_adus:
            r = self.edu_triples_to_adu_triples(r)
        # reduce to relation set
        reduced = {
            "exa": "sup",
            "add": "sup",
            "link": "sup",
            "join": "sup",
            "restate": "sup",
            "reb": "att",
            "und": "att",
        }
        if self.relation_set == SIMPLE_RELATION_SET:
            r = [(src, trg, reduced.get(rel, rel)) for src, trg, rel in r]
        # longer relation names
        longer = {
            "sup": "support",
            "exa": "example",
            "reb": "rebut",
            "und": "undercut",
            "att": "attack",
        }
        if long_names:
            r = [(src, trg, longer.get(rel, rel)) for src, trg, rel in r]
        # load from dependency triples
        self.load_from_triples(r, tid=g.graph.get("id"))

    def load_from_triples(self, triples, tid=None):
        """
        Loads a spanning tree from a list of [source, target, relationtype]
        triples.

        >>> triples = [[1, 2, 'sup'], [3, 2, 'att'], [4, 3, 'att']]
        >>> t = ArgTree()
        >>> t.load_from_triples(triples, tid='text-01')
        >>> t.graph['text']
        'text-01'
        >>> t.edges(data=True)
        [(1, 2, {'type': 'sup'}), (3, 2, {'type': 'att'}),
         (4, 3, {'type': 'att'})]
        """
        if tid is not None:
            self.graph["text"] = tid
        for src, trg, rel in triples:
            # harmonize linked claim relation name
            if rel == "add":
                rel = "link"
            if trg != 0:  # dont add artificial root node
                self.add_edge(src, trg, type=rel)

    def get_triples(self, include_root=False):
        """
        Returns a list of edges represented as [source, target, relationtype]
        triples.

        >>> t = ArgTree(from_triples=[(1,2,'sup'), (3,2,'att'), (4,3,'att')])
        >>> t.get_triples()
        [[1, 2, 'sup'], [3, 2, 'att'], [4, 3, 'att']]
        >>> t.get_triples(include_root=True)
        [[1, 2, 'sup'], [2, 'ROOT', 'ROOT'], [3, 2, 'att'], [4, 3, 'att']]
        """
        edges = [
            [src, trg, d["type"]] for src, trg, d in self.edges(data=True)
        ]
        if include_root:
            edges.append([self.get_cc(), "ROOT", "ROOT"])
        return sorted(edges)

    def get_tid(self):
        """
        Returns the text id name, if one is specified, else None.

        >>> t = ArgTree()
        >>> tid = t.get_tid()
        >>> tid is None
        True

        >>> t = ArgTree(text_id='text-01')
        >>> t.get_tid()
        'text-01'
        """
        return self.graph["text"] if "text" in self.graph else None

    def __str__(self):
        s = "<ArgTree:"
        s += " text_id={}".format(self.get_tid())
        s += " nodes={}".format(self.nodes())
        s += " edges={}>".format(self.get_triples())
        return s

    def get_cc(self):
        """
        Returns the central claim node, i.e. the root node, which has
        no outgoing edges.

        >>> t = ArgTree(from_triples=[(1,2,'sup'), (3,2,'att'), (4,3,'att')])
        >>> t.get_cc()
        2
        """
        _, ccnode = min([(self.out_degree(n), n) for n in self.nodes()])
        return ccnode

    def get_cc_vector(self):
        """
        Returns a vector as long as nodes in the tree. The vector position
        corresponding to the central claim node is 1, all others are 0.

        >>> t = ArgTree(from_triples=[(1,2,'sup'), (3,2,'att'), (4,3,'att')])
        >>> t.get_cc_vector()
        [0, 1, 0, 0]
        """
        r = [0] * len(self.nodes())
        cc = self.get_cc()
        r[cc - 1] = 1
        return r

    def get_ro_vector(self):
        """
        Returns a vector as long as nodes in the tree. The value at each
        vector position corresponds to is argumentative role, which is
        either 0 for proponent nodes, or 1 for opponent nodes.

        >>> t = ArgTree(from_triples=[(1,2,'sup'), (3,2,'att'), (4,3,'att')])
        >>> t.get_ro_vector()
        [0, 0, 1, 0]
        """
        # most complicated, traverse graph bottom-up, setting all roles
        # according to their type-labels
        r = [0] * len(self.nodes())
        map_role_to_vector = {"pro": 0, "opp": 1}
        queue = deque([(self.get_cc(), "pro")])
        nodes_seen = set([])
        while len(queue) > 0:
            node, role = queue.popleft()
            if node in nodes_seen:
                continue
            nodes_seen.add(node)
            r[node - 1] = map_role_to_vector[role]
            for src, _trg, d in self.in_edges(node, data=True):
                if d["type"] in self.relation_set.functions_inverting_role:
                    new_role = ArgTree.ROLES[role]
                else:
                    new_role = role
                queue.append((src, new_role))
        return r

    def get_fu_vector(self):
        """
        Returns a vector as long as nodes in the tree. The value at each
        vector position corresponds to is argumentative function defined
        in the relation set.

        >>> t = ArgTree(from_triples=[(1,2,'sup'), (3,2,'sup')])
        >>> t.get_fu_vector()
        [1, 0, 1]

        >>> t = ArgTree(from_triples=[(1,2,'sup'), (3,2,'att'), (4,3,'att')])
        >>> t.get_fu_vector()
        [1, 0, 2, 2]
        """
        function_map = self.relation_set.map_function_to_vector
        function_cc = self.relation_set.function_central_claim
        r = [function_map[function_cc]] * len(self.nodes())
        for (src, _trg, d) in sorted(self.edges(data=True)):
            r[src - 1] = function_map[d["type"]]
        return r

    def get_at_vector(self):
        """
        Returns a vector as long as all possible, non-identical node pairs.
        The value at each vector position corresponds to the fact, whether
        this edge exists in the tree: 0 for no attachment and 1 for attachment.

        >>> t = ArgTree(from_triples=[(1,2,'sup'), (3,2,'att'), (4,3,'att')])
        >>> t.get_at_vector()
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]
        """
        # all combinations, sorted
        combinations = sorted(list(permutations(self.nodes(), 2)))
        r = [0] * len(combinations)
        for i, (src, trg) in enumerate(combinations):
            if (src, trg) in self.edges([src]):
                r[i] = 1
        return r

    def get_vector(self):
        """
        Returns a dictionary with a vector for each level of the spanning
        tree: 'cc' for central claim, 'ro' for argumentative role, 'fu' for
        argumentative function and 'at' for attachment.

        >>> t = ArgTree(from_triples=[(1,2,'sup'), (3,2,'att'), (4,3,'att')])
        >>> sorted(t.get_vector().items())
        [('at', [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]), ('cc', [0, 1, 0, 0]),
         ('fu', [1, 0, 2, 2]), ('ro', [0, 0, 1, 0])]

        >>> t = ArgTree(relation_set=FULL_RELATION_SET, from_triples=[(1,2,'support'), (3,2,'rebut'), (4,3,'link')])
        >>> sorted(t.get_vector().items())
        [('at', [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]), ('cc', [0, 1, 0, 0]),
         ('fu', [1, 0, 5, 4]), ('ro', [0, 0, 1, 1])]
        """
        return {
            "cc": self.get_cc_vector(),
            "ro": self.get_ro_vector(),
            "fu": self.get_fu_vector(),
            "at": self.get_at_vector(),
        }

    def get_folding_label(self):
        """
        Returns a label which can be used as a reference when sampling a
        representative folding.

        >>> t = ArgTree(from_triples=[(1,2,'sup'), (3,2,'att'), (4,3,'att')])
        >>> t.get_folding_label()
        ['01', '00', '12', '02']
        """
        return [
            str(ro) + str(fu)
            for ro, fu in zip(self.get_ro_vector(), self.get_fu_vector())
        ]
