#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: Andreas Peldszus
"""
from __future__ import print_function

import re
import networkx as nx
from lxml import etree
from Queue import Queue
from textwrap import wrap
from pydot import graph_from_dot_data


def sorted_nicely(l):
    """
    Sorts the given iterable in the way that is expected.
    http://stackoverflow.com/a/2669120

    Args:
        l (iterable): The iterable to be sorted.

    Returns:
        a sorted iterable

    >>> sorted_nicely(['A100', 'A10', 'A1.1', 'A2', 'A1', 'Z', 'ZZ'])
    ['A1', 'A1.1', 'A2', 'A10', 'A100', 'Z', 'ZZ']
    """

    def convert(text):
        return int(text) if text.isdigit() else text

    def alphanum_key(key):
        return [convert(c) for c in re.split("([0-9]+)", key)]

    return sorted(l, key=alphanum_key)


class ArgGraphException(nx.NetworkXException):
    """ A class for exceptions raised in the handling of argumentation
        graphs """


class ArgGraph(nx.DiGraph):

    ROLES = ["pro", "opp"]

    def add_edu(self, edu_id, edu_txt):
        """
        >>> g = ArgGraph()
        >>> g.add_edu('e1', 'This is an EDU.')
        >>> g.node == {'e1': {'type': 'edu', 'text': 'This is an EDU.'}}
        True
        """
        self.add_node(edu_id, type="edu", text=edu_txt)

    def add_edus(self, edus, start_with_id=1):
        """
        >>> g = ArgGraph()
        >>> g.add_edus(['We.', 'Should.', 'Swim!'], start_with_id=2)
        >>> g.node == {'e2': {'type': 'edu', 'text': 'We.'}, 'e3': {'type': 'edu', 'text': 'Should.'}, 'e4': {'type': 'edu', 'text': 'Swim!'}}
        True
        """
        for i, edu in enumerate(edus, start_with_id):
            self.add_edu("e{}".format(i), edu)

    def add_edu_joint(self, joint_id, edus=None):
        """
        >>> g = ArgGraph()
        >>> g.add_edu_joint('j1')
        >>> g.node == {'j1': {'type': 'joint'}}
        True

        >>> g = ArgGraph()
        >>> g.add_edu('e1', 'EDU 1')
        >>> g.add_edu('e2', 'EDU 2')
        >>> g.add_edu_joint('j1', ['e1', 'e2'])
        >>> g.node == {'e1': {'type': 'edu', 'text': 'EDU 1'}, 'e2': {'type': 'edu', 'text': 'EDU 2'}, 'j1': {'type': 'joint'}}
        True
        >>> g.edge == {'e1': {'j1': {'type': 'seg'}}, 'e2': {'j1': {'type': 'seg'}}, 'j1': {}}
        True
        """
        self.add_node(joint_id, type="joint")
        if edus:
            for edu in edus:
                self.add_seg_edge(edu, joint_id)

    def add_adu(self, adu_id, adu_role):
        """
        >>> g = ArgGraph()
        >>> g.add_adu('a1', 'pro')
        >>> g.node == {'a1': {'type': 'adu', 'role': 'pro'}}
        True
        >>> g.add_adu('a2', 'unknown')
        Traceback (most recent call last):
        [...]
        AssertionError: Undefined role.
        """
        assert adu_role in ArgGraph.ROLES, "Undefined role."
        self.add_node(adu_id, type="adu", role=adu_role)

    def add_seg_edge(self, edge_src, edge_trg):
        """
        >>> g = ArgGraph()
        >>> g.add_edu('e1', 'EDU 1')
        >>> g.add_adu('a1', 'pro')
        >>> g.add_seg_edge('e1', 'a1')
        >>> g.edge == {'e1': {'a1': {'type': 'seg'}}, 'a1': {}}
        True
        """
        assert edge_src in self.node, "Source node does not exist in graph"
        assert edge_trg in self.node, "Target node does not exist in graph"
        self.add_edge(edge_src, edge_trg, type="seg")

    def is_adu(self, id_, role=None):
        if id_ not in self.node:
            return False
        elif "type" not in self.node[id_] or self.node[id_]["type"] != "adu":
            return False
        elif role:
            try:
                return self.node[id_]["role"] == role
            except KeyError:
                return False
        else:
            return True

    def add_edu_adu(self, edu_id, edu_txt, adu_id, adu_role):
        self.add_edu(edu_id, edu_txt)
        self.add_adu(adu_id, adu_role)
        self.add_seg_edge(edu_id, adu_id)

    def add_edge_with_relation_node(
        self, edge_id, edge_src, edge_trg, edge_type
    ):
        self.add_node(edge_id, type="rel")
        self.add_edge(edge_src, edge_id, type="src")
        self.add_edge(edge_id, edge_trg, type=edge_type)

    def load_from_xml(self, filename):
        xml = etree.parse(filename)

        # graph id
        text_id = xml.xpath("/arggraph")[0].get("id")
        self.graph["id"] = text_id

        # add all EDU
        for elm in xml.xpath("/arggraph/edu"):
            self.add_edu(elm.get("id"), elm.text)
        # add all EDU-JOINS
        for elm in xml.xpath("/arggraph/joint"):
            self.add_edu_joint(elm.get("id"))
        # add all ADU
        for elm in xml.xpath("/arggraph/adu"):
            self.add_adu(elm.get("id"), elm.get("type"))

        # add all edges
        q = Queue()
        for elm in xml.xpath("/arggraph/edge"):
            q.put(elm)
        while not q.empty():
            # TODO: queue processing might not end for input elements with
            #       malformed targets, cyclic relations
            elm = q.get()

            edge_src = elm.get("src")
            if edge_src not in self.nodes():
                print ("Error: source unknown\n", etree.tostring(elm))

            edge_trg = elm.get("trg")
            if edge_trg not in self.nodes():
                # target node (of 'und' or 'add' relations) not there yet.
                # postpone to later
                q.put(elm)
                continue

            edge_type = elm.get("type")
            edge_id = elm.get("id")
            if edge_type == "seg":
                src_trg_type_pair = (
                    self.node[edge_src]["type"],
                    self.node[edge_trg]["type"],
                )
                if src_trg_type_pair in [
                    ("edu", "adu"),
                    ("edu", "joint"),
                    ("joint", "adu"),
                ]:
                    self.add_seg_edge(edge_src, edge_trg)
                else:
                    print (
                        "Error: malformed segmentation edge\n",
                        etree.tostring(elm),
                    )

            elif edge_type in ["sup", "exa", "reb"]:
                if (
                    self.node[edge_src]["type"] == "adu"
                    and self.node[edge_trg]["type"] == "adu"
                ):
                    self.add_edge_with_relation_node(
                        edge_id, edge_src, edge_trg, edge_type
                    )
                else:
                    print (
                        "Error: malformed direct edge\n",
                        etree.tostring(elm),
                    )

            elif edge_type == "und":
                if (
                    self.node[edge_src]["type"] == "adu"
                    and self.node[edge_trg]["type"] == "rel"
                ):
                    self.add_edge_with_relation_node(
                        edge_id, edge_src, edge_trg, edge_type
                    )
                else:
                    print (
                        (
                            "Error: malformed undercutting edge\n",
                            etree.tostring(elm),
                        )
                    )

            elif edge_type == "add":
                if (
                    self.node[edge_src]["type"] == "adu"
                    and self.node[edge_trg]["type"] == "rel"
                ):
                    self.add_edge(elm.get("src"), elm.get("trg"), type="src")
                else:
                    print (
                        "Error: malformed adding edge\n",
                        etree.tostring(elm),
                    )

            else:
                print ("Error: unknown edge type\n", etree.tostring(elm))

        # update adu short names
        self.update_adu_labels()

    def to_xml(self):
        """
        >>> a = get_complex_arggraph()
        >>> print(a.to_xml())
        <?xml version='1.0' encoding='UTF-8'?>
        <arggraph id="g1">
          <edu id="e1"><![CDATA[Swim!]]></edu>
          <edu id="e2"><![CDATA[Good weather.]]></edu>
          <edu id="e3"><![CDATA[Sharks!]]></edu>
          <edu id="e4"><![CDATA[!!!1!]]></edu>
          <edu id="e5"><![CDATA[Bought anti-sharks-spray.]]></edu>
          <edu id="e6"><![CDATA[It is effective.]]></edu>
          <edu id="e7"><![CDATA[So let us swim!]]></edu>
          <joint id="j1"/>
          <adu id="a1" type="pro"/>
          <adu id="a2" type="pro"/>
          <adu id="a3" type="opp"/>
          <adu id="a4" type="pro"/>
          <adu id="a5" type="pro"/>
          <edge id="c1" src="e1" trg="a1" type="seg"/>
          <edge id="c2" src="e2" trg="a2" type="seg"/>
          <edge id="c3" src="e3" trg="j1" type="seg"/>
          <edge id="c4" src="e4" trg="j1" type="seg"/>
          <edge id="c5" src="e5" trg="a4" type="seg"/>
          <edge id="c6" src="e6" trg="a5" type="seg"/>
          <edge id="c7" src="e7" trg="a1" type="seg"/>
          <edge id="c8" src="j1" trg="a3" type="seg"/>
          <edge id="c9" src="a2" trg="a1" type="sup"/>
          <edge id="c10" src="a3" trg="c9" type="und"/>
          <edge id="c11" src="a4" trg="c10" type="und"/>
          <edge id="c12" src="a5" trg="c11" type="add"/>
        </arggraph>

        """
        # xml serialization only for non-transformed graphs
        if (
            "relation-node-free" in self.graph
            and self.graph["relation-node-free"] == True
        ):
            return None

        edge_xml = '<edge id="{}" src="{}" trg="{}" type="{}" />'

        # try to find source text
        doc_elm = etree.XML('<arggraph id="{}"/>'.format(self.graph["id"]))

        # a mapping from node_ids to xml_ids
        new_ids = {}

        # serialize edus
        max_edu_count = 1
        edus = self.get_edus()
        for node in sorted_nicely(edus.keys()):
            edu_text = edus[node]
            new_edu_id = "e{}".format(max_edu_count)
            new_ids[node] = new_edu_id
            max_edu_count += 1
            edu_elm = etree.XML('<edu id="{}" />'.format(new_edu_id))
            edu_elm.text = etree.CDATA(edu_text)
            doc_elm.append(edu_elm)

        # serialize joints
        max_joint_count = 1
        for node in sorted_nicely(self.get_joints()):
            joint_id = "j{}".format(max_joint_count)
            new_ids[node] = joint_id
            max_joint_count += 1
            joint_elm = etree.XML('<joint id="{}" />'.format(joint_id))
            doc_elm.append(joint_elm)

        # serialize adus
        max_adu_count = 1
        for node in sorted_nicely(self.get_adus()):
            adu_type = self.node[node]["role"]
            adu_id = "a{}".format(max_adu_count)
            new_ids[node] = adu_id
            max_adu_count += 1
            adu_elm = etree.XML(
                '<adu id="{}" type="{}" />'.format(adu_id, adu_type)
            )
            doc_elm.append(adu_elm)

        # serialize edges
        edge_elms = []
        max_edge_count = 1

        # segmentation edges
        for source, target, data in sorted(self.edges(data=True)):
            edge_type = data["type"]
            node_types = (self.node[source]["type"], self.node[target]["type"])
            if edge_type == "seg" and node_types in [
                ("edu", "adu"),
                ("edu", "joint"),
                ("joint", "adu"),
            ]:
                edge_id = "c%d" % max_edge_count
                max_edge_count += 1
                edge_elm = etree.XML(
                    edge_xml.format(
                        edge_id, new_ids[source], new_ids[target], "seg"
                    )
                )
                edge_elms.append(edge_elm)

        # iterate over (multi source) adu adu edges by relation nodes
        q = Queue()
        for node, data in self.nodes(data=True):
            if data["type"] == "rel":
                q.put(node)
        while not q.empty():
            node = q.get()
            target = self.successors(node)[0]
            if target not in new_ids:
                # we do not know the new id of the target yet (might be another relation node)
                q.put(node)
                continue
            # go on
            sources = sorted_nicely(
                [
                    source
                    for source in self.predecessors(node)
                    if (
                        self.node[source]["type"] == "adu"
                        and self.edge[source][node]["type"] == "src"
                    )
                ]
            )
            # first source
            source = sources[0]
            edge_type = self.edge[node][target]["type"]
            edge_id = "c%d" % max_edge_count
            max_edge_count += 1
            new_ids[node] = edge_id
            edge_elm = etree.XML(
                edge_xml.format(
                    edge_id, new_ids[source], new_ids[target], edge_type
                )
            )
            edge_elms.append(edge_elm)
            to_edge_id = edge_id
            # rest sources
            for source in sources[1:]:
                edge_id = "c%d" % max_edge_count
                max_edge_count += 1
                edge_elm = etree.XML(
                    edge_xml.format(
                        edge_id, new_ids[source], to_edge_id, "add"
                    )
                )
                edge_elms.append(edge_elm)

        for e in edge_elms:
            doc_elm.append(e)

        return etree.tostring(
            doc_elm, encoding="UTF-8", pretty_print=True, xml_declaration=True
        )

    def update_adu_labels(self):
        # first label all edus
        for edu_node in [
            i for i, d in self.nodes(data=True) if d["type"] == "edu"
        ]:
            self.node[edu_node]["nr-label"] = edu_node.replace("e", "")

        # then all joints
        for joint_node in [
            i for i, d in self.nodes(data=True) if d["type"] == "joint"
        ]:
            label = "+".join(
                sorted_nicely(
                    [
                        self.node[i]["nr-label"]
                        for i in self.predecessor_by_edge_type(
                            joint_node, "seg"
                        )
                    ]
                )
            )
            self.node[joint_node]["nr-label"] = label

        # then all adu
        for adu_node in [
            i for i, d in self.nodes(data=True) if d["type"] == "adu"
        ]:
            label = "=".join(
                sorted_nicely(
                    [
                        self.node[i]["nr-label"]
                        for i in self.predecessor_by_edge_type(adu_node, "seg")
                    ]
                )
            )
            self.node[adu_node]["nr-label"] = label

    def predecessors_with_node_type(self, node, node_type):
        return [
            i
            for i in self.predecessors(node)
            if "type" in self.node[i] and self.node[i]["type"] == type
        ]

    def predecessor_by_edge_type(self, node, edge_type):
        return [
            src
            for src, trg, d in self.edges(data=True)
            if trg == node and "type" in d and d["type"] == edge_type
        ]

    def get_relation_node_free_graph(self):
        if nx.is_strongly_connected(self):
            raise ArgGraphException(
                (
                    "Cannot produce relation node free graph."
                    "Arggraph contains cycles."
                )
            )
        if False in [self.out_degree(node) <= 1 for node in self.nodes()]:
            raise ArgGraphException(
                (
                    "Cannot produce relation node free graph."
                    "Nodes with multiple outgoing edges."
                )
            )

        a = ArgGraph(self)

        if (
            "relation-node-free" in a.graph
            and a.graph["relation-node-free"] == True
        ):
            return a

        # reduce multi-source relations to adu.addsource->adu
        for rel_node in [
            node
            for node, d in a.nodes(data=True)
            if a.out_degree(node) >= 1 and d["type"] == "rel"
        ]:
            sources = sorted_nicely(
                [
                    source
                    for source in a.predecessors(rel_node)
                    if a.node[source]["type"] == "adu"
                ]
            )
            for source in sources[1:]:
                a.remove_edge(source, rel_node)
                a.add_edge(source, sources[0], type="add")

        # first reduce rel->rel
        remove_nodes = []
        remove_edges = []
        for (src, trg, d) in a.edges(data=True):
            if a.node[src]["type"] == "rel" and a.node[trg]["type"] == "rel":
                src_pre = a.predecessor_by_edge_type(src, "src")[0]
                trg_pre = a.predecessor_by_edge_type(trg, "src")[0]
                a.remove_edge(src, trg)
                a.add_edge(src_pre, trg_pre, type=d["type"])
                remove_edges.append((src_pre, src))
                remove_nodes.append(src)

        for src, trg in remove_edges:
            a.remove_edge(src, trg)
        for node in remove_nodes:
            a.remove_node(node)

        # then reduce rel->adu (remaining relnodes)
        for (src, trg, d) in a.edges(data=True):
            if a.node[src]["type"] == "rel" and a.node[trg]["type"] == "adu":
                src_pre = a.predecessors(src)[0]
                a.add_edge(src_pre, trg, type=d["type"])
                a.remove_edge(src_pre, src)
                a.remove_edge(src, trg)
                a.remove_node(src)

        a.graph["relation-node-free"] = True

        return a

    def export_to_dot(self, edu_cluster=False, wrap_width=20):
        queries_nodes = {
            "edu": lambda d: d["type"] == "edu",
            "joint": lambda d: d["type"] == "joint",
            "pro": lambda d: d["type"] == "adu" and d["role"] == "pro",
            "opp": lambda d: d["type"] == "adu" and d["role"] == "opp",
            "rel": lambda d: d["type"] == "rel",
        }

        styles_nodes = {
            "edu": (
                '[shape=box, style=filled, color="#aaaaaa", '
                'fillcolor="#f0f0f0", fontsize=10, width=1.0];'
                "\nstyle=invis;"
            ),
            "joint": (
                '[shape=box, style=filled, color="#aaaaaa", '
                'fillcolor="#f0f0f0"];'
            ),
            "pro": "[shape=oval];",
            "opp": "[shape=rect];",
            "rel": (
                '[shape=octagon, style=filled, color="#aaaaaa", '
                "fixedsize=true, width=0.3, height=0.3, "
                'fillcolor="#FFF8DC", fontsize=10];'
            ),
        }

        styles_edges = {
            "seg": '[weight=1, arrowhead=none, color="#aaaaaa"];',
            "src": "[weight=1, arrowhead=none];",
            "sup": "[weight=1, arrowhead=open];",
            "exa": "[weight=1, arrowhead=open, style=dashed];",
            "reb": "[weight=1, arrowhead=dot];",
            "und": "[weight=1, arrowhead=box];",
            "add": "[weight=1, arrowhead=empty];",
        }

        # TODO:
        # When plotting the graph TB (thus linearizing EDUS left to right)
        # crossing edges work.
        # When plotting the graph LR (thus linearizing EDUS top down)
        # crossing edges don't work.

        # template_graph = u'digraph G {\n// %s\nrankdir=LR\n%s}' # name content  # noqa
        template_graph = u"digraph G {\n// %s\n%s}"  # name content
        template_subgraph = u"subgraph %s {\n%s\n}"  # name content
        template_node_label = u'%s [label="%s"];'  # id label

        edu_content = u""

        # first edus nodes
        node_type = "edu"
        data = "node " + styles_nodes[node_type]
        data += "\nrank=same;"
        data += "\nrankdir=TB;"
        nodes = sorted_nicely(
            [
                i
                for (i, d) in self.nodes(data=True)
                if queries_nodes[node_type](d)
            ]
        )
        for i in nodes:
            text = self.node[i]["text"].replace('"', "''")
            wrapped_text = wrap("[%s] " % i + text, width=wrap_width)
            data += "\n" + template_node_label % (i, "\\n".join(wrapped_text))

        # add invisible edges from edu to edu to enforce linearity in the graph
        # template_linearity_subgraph = '\nsubgraph linearity {\nedge [weight=8, color="#ffcccc"];\n%s}'  # noqa
        template_linearity_subgraph = (
            "\nsubgraph linearity {\nedge [weight=8, style=invis];\n%s}"
        )  # noqa
        edges = ""
        for src, trg in zip(nodes, nodes[1:]):
            edges += "%s -> %s\n" % (src, trg)
        linearity_subgraph = template_linearity_subgraph % edges

        if edu_cluster:
            subgraph = template_subgraph % (
                "cluster_" + node_type,
                data + linearity_subgraph,
            )
        else:
            subgraph = template_subgraph % (
                node_type,
                data + linearity_subgraph,
            )
        edu_content += "\n" + subgraph

        graph_content = u""

        # remaining nodes
        for node_type in styles_nodes:
            if node_type == "edu":
                continue
            data = ""
            data += "node " + styles_nodes[node_type]
            nodes = [
                (i, d)
                for (i, d) in self.nodes(data=True)
                if queries_nodes[node_type](d)
            ]
            for i, d in nodes:
                node_label = i
                if "nr-label" in self.node[i]:
                    node_label = self.node[i]["nr-label"]
                data += "\n" + template_node_label % (i, node_label)
            graph_content += "\n" + template_subgraph % (node_type, data)

        # edges
        for edge_type in styles_edges:
            data = ""
            data += "edge " + styles_edges[edge_type]
            edges = [
                (src, trg, d)
                for (src, trg, d) in self.edges(data=True)
                if d["type"] == edge_type
            ]
            for src, trg, _ in edges:
                data += "\n" + src + " -> " + trg
            graph_content += "\n" + template_subgraph % (edge_type, data)

        content = (
            edu_content
            + "\n"
            + template_subgraph % ("graph_rest", graph_content)
        )

        dot = template_graph % ("automatically generated", content)
        return dot

    def show_in_ipynb(self, edu_cluster=False):
        from IPython.display import Image

        dot = self.export_to_dot(edu_cluster=edu_cluster).encode("utf-8")
        return Image(graph_from_dot_data(dot).create_png())

    def render_as_dot(self, edu_cluster=False):
        dot = self.export_to_dot(edu_cluster=edu_cluster)
        dot_utf8 = dot.encode("utf-8")
        return dot_utf8

    def render_as_png(self, filename, edu_cluster=False):
        dot_utf8 = self.render_as_dot(edu_cluster=edu_cluster)
        dot_graph = graph_from_dot_data(dot_utf8)
        dot_graph.write_png(filename)

    def render_as_pdf(self, filename, edu_cluster=False):
        dot_utf8 = self.render_as_dot(edu_cluster=edu_cluster)
        dot_graph = graph_from_dot_data(dot_utf8)
        dot_graph.write_pdf(filename)

    def get_edus(self):
        return {
            i: d["text"]
            for (i, d) in self.nodes(data=True)
            if d["type"] == "edu"
        }

    def get_joints(self):
        return {i for (i, d) in self.nodes(data=True) if d["type"] == "joint"}

    def get_adus(self):
        return {i for (i, d) in self.nodes(data=True) if d["type"] == "adu"}

    def get_edus_of_adu(self, adu):
        """
        >>> g = get_very_complex_arggraph()
        >>> g.get_edus_of_adu('a1')
        [('e1', 'txt1'), ('e2', 'txt2'), ('e7', 'txt7')]
        """
        assert adu in self.node and self.node[adu]["type"] == "adu"
        worklist, visited, edus = set([adu]), set([]), set([])
        while worklist:
            current = worklist.pop()
            visited.add(current)
            if self.node[current]["type"] == "edu":
                edus.add(current)
            for pre in self.predecessor_by_edge_type(current, "seg"):
                worklist.add(pre)
        return [(edu, self.node[edu]["text"]) for edu in sorted_nicely(edus)]

    def get_adu_segmented_text_with_restatements(self):
        """
        >>> g = get_very_complex_arggraph()
        >>> g.get_adu_segmented_text_with_restatements()
        ['txt1 txt2', 'txt3 txt4', 'txt5', 'txt6', 'txt7']
        """
        edu_triples = self.get_edus_as_dependencies(
            include_cc=True, ids_to_numbers=True
        )
        edu_segments = self.get_segmented_text()
        adu_segments = []
        for (_src, _trg, rel), txt in zip(edu_triples, edu_segments):
            if rel == "join":
                adu_segments[-1] += " " + txt
            else:
                adu_segments.append(txt)
        return adu_segments

    def get_adu_segmented_text(self):
        """
        >>> g = get_very_complex_arggraph()
        >>> g.get_adu_segmented_text()
        ['txt1 txt2 txt7', 'txt3 txt4', 'txt5', 'txt6']
        """
        return [
            " ".join([text for _, text in self.get_edus_of_adu(adu)])
            for adu in sorted_nicely(self.get_adus())
        ]

    def get_segmented_text(self):
        """
        >>> g = get_very_complex_arggraph()
        >>> g.get_segmented_text()
        ['txt1', 'txt2', 'txt3', 'txt4', 'txt5', 'txt6', 'txt7']
        """
        edus = self.get_edus()
        return [edus[i] for i in sorted_nicely(edus.keys())]

    def get_unsegmented_text(self):
        """
        >>> g = get_very_complex_arggraph()
        >>> g.get_unsegmented_text()
        'txt1 txt2 txt3 txt4 txt5 txt6 txt7'
        """
        return " ".join(self.get_segmented_text())

    def get_adu_adu_relations(self):
        """
        >>> g = get_very_complex_arggraph()
        >>> g.get_adu_adu_relations()
        [('a3', 'a2', 'add'), ('a2', 'a1', 'reb'), ('a4', 'a2', 'und')]
        """
        # make sure to extract relations from a relation node free graph
        if (
            "relation-node-free" in self.graph
            and self.graph["relation-node-free"] == True
        ):
            g = self
        else:
            g = self.get_relation_node_free_graph()

        # get all adu-adu relations
        return [
            (src, trg, d["type"])
            for src, trg, d in g.edges(data=True)
            if (g.node[src]["type"] == "adu" and g.node[trg]["type"] == "adu")
        ]

    def get_adus_as_dependencies(self, include_cc=False, ids_to_numbers=False):
        """
        >>> g = get_minimal_arggraph()
        >>> g.get_adus_as_dependencies()
        [('a2', 'a1', 'sup')]
        >>> g.get_adus_as_dependencies(include_cc=True)
        [('a1', 'a0', 'ROOT'), ('a2', 'a1', 'sup')]
        >>> g.get_adus_as_dependencies(include_cc=True, ids_to_numbers=True)
        [(1, 0, 'ROOT'), (2, 1, 'sup')]

        >>> g = get_arggraph_restatement()
        >>> g.get_adus_as_dependencies(include_cc=True, ids_to_numbers=True)
        [(1, 0, 'ROOT'), (2, 1, 'sup')]

        >>> g = get_arggraph_joint()
        >>> g.get_adus_as_dependencies(include_cc=True, ids_to_numbers=True)
        [(1, 0, 'ROOT'), (2, 1, 'sup')]

        >>> g = get_very_complex_arggraph()
        >>> g.get_adus_as_dependencies(include_cc=True, ids_to_numbers=True)
        [(1, 0, 'ROOT'), (2, 1, 'reb'), (3, 2, 'add'), (4, 2, 'und')]

        """

        def a2id(s):
            return int(s[1:])

        r = self.get_adu_adu_relations()
        if include_cc:
            r.append((self.get_central_claim(), "a0", "ROOT"))
        if ids_to_numbers:
            r = [(a2id(src), a2id(trg), rel) for src, trg, rel in r]
        return sorted(r)

    def get_edus_as_dependencies(self, include_cc=False, ids_to_numbers=False):
        """
        >>> g = get_minimal_arggraph()
        >>> g.get_edus_as_dependencies(ids_to_numbers=True)
        [(2, 1, 'sup')]

        >>> g = get_simple_arggraph()
        >>> g.get_edus_as_dependencies(ids_to_numbers=True)
        [(2, 1, 'sup'), (3, 2, 'und'), (4, 3, 'und')]

        >>> g = get_arggraph_restatement()
        >>> g.get_edus_as_dependencies(ids_to_numbers=True)
        [(2, 1, 'sup'), (3, 1, 'restate')]

        >>> g = get_arggraph_joint()
        >>> g.get_edus_as_dependencies(ids_to_numbers=True)
        [(2, 1, 'join'), (3, 1, 'join'), (4, 1, 'sup')]

        >>> g = get_arggraph_linked()
        >>> g.get_edus_as_dependencies(ids_to_numbers=True)
        [(2, 1, 'sup'), (3, 2, 'link')]

        >>> g = get_very_complex_arggraph()
        >>> g.get_edus_as_dependencies(ids_to_numbers=True)
        [(2, 1, 'join'), (3, 1, 'reb'), (4, 3, 'join'), (5, 3, 'link'),
         (6, 3, 'und'), (7, 1, 'restate')]

        >>> g = get_simple_arggraph()
        >>> g.get_edus_as_dependencies(include_cc=True, ids_to_numbers=True)
        [(1, 0, 'ROOT'), (2, 1, 'sup'), (3, 2, 'und'), (4, 3, 'und')]

        >>> g = get_simple_arggraph()
        >>> g.get_edus_as_dependencies(ids_to_numbers=False)
        [('e2', 'e1', 'sup'), ('e3', 'e2', 'und'), ('e4', 'e3', 'und')]
        """

        def e2id(s):
            return int(s[1:])

        r = []

        head = {edu: edu for edu in self.get_edus().keys()}

        # direct segmentation edges (edu to adu)
        adus = self.get_adus()
        for adu in adus:
            edus = [
                i
                for i in self.predecessors(adu)
                if self.node[i]["type"] == "edu"
            ]
            if len(edus) == 1:
                head[adu] = edus[0]

        # low level segmentation edges (from edus to joints)
        for joint in self.get_joints():
            edus = sorted_nicely(self.predecessors(joint))
            first = edus[0]
            head[joint] = first
            for edu in edus[1:]:
                r.append((edu, first, "join"))

        # higher level segmentation edges (from joints or edus to adus)
        for adu in adus:
            edus = sorted_nicely(
                [
                    head[i]
                    for i in self.predecessors(adu)
                    if self.node[i]["type"] in ["edu", "joint"]
                ]
            )
            first = edus[0]
            head[adu] = first
            for edu in edus[1:]:
                r.append((edu, first, "restate"))

        # argumentation relations (from adu to adu)
        q = Queue()
        for node, data in self.nodes(data=True):
            if data["type"] == "rel":
                q.put(node)
        while not q.empty():
            node = q.get()
            target = self.successors(node)[0]
            if target not in head:
                q.put(node)
                continue
            edus = sorted_nicely(
                [
                    head[i]
                    for i in self.predecessors(node)
                    if self.node[i]["type"] == "adu"
                ]
            )
            first = edus[0]
            head[node] = first
            r.append((first, head[target], self.edge[node][target]["type"]))
            for edu in edus[1:]:
                r.append((edu, first, "link"))

        # covert and then sort
        if include_cc:
            r.append((self.get_central_claim(), "e0", "ROOT"))
        if ids_to_numbers:
            r = [(e2id(src), e2id(trg), rel) for src, trg, rel in r]
        return sorted(r)

    def get_adu_role(self, adu):
        return self.node[adu]["role"]

    def get_adu_functions(self, adu):  # todo central claim
        if (
            "relation-node-free" in self.graph
            and self.graph["relation-node-free"] == True
        ):
            return self.edges(adu, data=True)
        else:
            # outgoing arcs go to relation nodes first, we want the arc _from_
            # that relation node
            out = self.edges(adu)
            if len(out) == 0:
                return []
            else:
                relnode = self.edges(adu)[0][1]
                return self.edges(relnode, data=True)

    def get_central_claim(self):
        _outdegree, ccnode = min(
            [(self.out_degree(n), n) for n in self.nodes()]
        )
        return ccnode

    def get_role_type_labels(self):
        aar = self.get_adu_adu_relations()
        return {
            src: "%s+%s" % (self.node[src]["role"], dtype)
            for src, _trg, dtype in aar
        }


def get_minimal_arggraph():
    """
    >>> a = get_minimal_arggraph()
    >>> a.node
    {'a1': {'role': 'pro', 'type': 'adu'}, 'c1': {'type': 'rel'}, 'a2': {'role': 'pro', 'type': 'adu'}, 'e1': {'text': 'Swim!', 'type': 'edu'}, 'e2': {'text': 'Good weather.', 'type': 'edu'}}
    >>> a.edge
    {'a1': {}, 'c1': {'a1': {'type': 'sup'}}, 'a2': {'c1': {'type': 'src'}}, 'e1': {'a1': {'type': 'seg'}}, 'e2': {'a2': {'type': 'seg'}}}
    """
    a = ArgGraph(id="g1")
    a.add_edu_adu("e1", "Swim!", "a1", "pro")
    a.add_edu_adu("e2", "Good weather.", "a2", "pro")
    a.add_edge_with_relation_node("c1", "a2", "a1", "sup")
    return a


def get_simple_arggraph():
    """
    >>> a = get_simple_arggraph()
    >>> a.node
    {'a1': {'role': 'pro', 'type': 'adu'}, 'a3': {'role': 'opp', 'type': 'adu'}, 'a2': {'role': 'pro', 'type': 'adu'}, 'a4': {'role': 'pro', 'type': 'adu'}, 'e4': {'text': 'Bought anti-sharks-spray.', 'type': 'edu'}, 'c3': {'type': 'rel'}, 'c2': {'type': 'rel'}, 'c1': {'type': 'rel'}, 'e1': {'text': 'Swim!', 'type': 'edu'}, 'e3': {'text': 'Sharks!!!!1!', 'type': 'edu'}, 'e2': {'text': 'Good weather.', 'type': 'edu'}}
    >>> a.edge
    {'a1': {}, 'a3': {'c2': {'type': 'src'}}, 'a2': {'c1': {'type': 'src'}}, 'a4': {'c3': {'type': 'src'}}, 'e4': {'a4': {'type': 'seg'}}, 'c3': {'c2': {'type': 'und'}}, 'c2': {'c1': {'type': 'und'}}, 'c1': {'a1': {'type': 'sup'}}, 'e1': {'a1': {'type': 'seg'}}, 'e3': {'a3': {'type': 'seg'}}, 'e2': {'a2': {'type': 'seg'}}}
    """
    a = ArgGraph(id="g1")
    a.add_edu_adu("e1", "Swim!", "a1", "pro")
    a.add_edu_adu("e2", "Good weather.", "a2", "pro")
    a.add_edge_with_relation_node("c1", "a2", "a1", "sup")
    a.add_edu_adu("e3", "Sharks!!!!1!", "a3", "opp")
    a.add_edge_with_relation_node("c2", "a3", "c1", "und")
    a.add_edu_adu("e4", "Bought anti-sharks-spray.", "a4", "pro")
    a.add_edge_with_relation_node("c3", "a4", "c2", "und")
    return a


def get_complex_arggraph():
    """
    >>> a = get_complex_arggraph()
    >>> a.node
    {'E6': {'text': 'So let us swim!', 'type': 'edu'}, 'J1': {'type': 'joint'}, 'A1': {'role': 'pro', 'type': 'adu'}, 'E5': {'text': 'It is effective.', 'type': 'edu'}, 'A3': {'role': 'opp', 'type': 'adu'}, 'A2': {'role': 'pro', 'type': 'adu'}, 'A5': {'role': 'pro', 'type': 'adu'}, 'A4': {'role': 'pro', 'type': 'adu'}, 'E4': {'text': 'Bought anti-sharks-spray.', 'type': 'edu'}, 'C3': {'type': 'rel'}, 'C2': {'type': 'rel'}, 'C1': {'type': 'rel'}, 'E3.1': {'text': '!!!1!', 'type': 'edu'}, 'E1': {'text': 'Swim!', 'type': 'edu'}, 'E3': {'text': 'Sharks!', 'type': 'edu'}, 'E2': {'text': 'Good weather.', 'type': 'edu'}}
    >>> a.edge
    {'E6': {'A1': {'type': 'seg'}}, 'J1': {'A3': {'type': 'seg'}}, 'A1': {}, 'E5': {'A5': {'type': 'seg'}}, 'A3': {'C2': {'type': 'src'}}, 'A2': {'C1': {'type': 'src'}}, 'A5': {'C3': {'type': 'src'}}, 'A4': {'C3': {'type': 'src'}}, 'E4': {'A4': {'type': 'seg'}}, 'C3': {'C2': {'type': 'und'}}, 'C2': {'C1': {'type': 'und'}}, 'C1': {'A1': {'type': 'sup'}}, 'E3.1': {'J1': {'type': 'seg'}}, 'E1': {'A1': {'type': 'seg'}}, 'E3': {'J1': {'type': 'seg'}}, 'E2': {'A2': {'type': 'seg'}}}
    """
    a = ArgGraph(id="g1")
    a.add_edu_adu("E1", "Swim!", "A1", "pro")
    a.add_edu_adu("E2", "Good weather.", "A2", "pro")
    a.add_edge_with_relation_node("C1", "A2", "A1", "sup")
    a.add_edu_joint("J1")
    a.add_edu("E3", "Sharks!")
    a.add_edge("E3", "J1", type="seg")
    a.add_edu("E3.1", "!!!1!")
    a.add_edge("E3.1", "J1", type="seg")
    a.add_adu("A3", "opp")
    a.add_edge("J1", "A3", type="seg")
    a.add_edge_with_relation_node("C2", "A3", "C1", "und")
    a.add_edu_adu("E4", "Bought anti-sharks-spray.", "A4", "pro")
    a.add_edge_with_relation_node("C3", "A4", "C2", "und")
    a.add_edu_adu("E5", "It is effective.", "A5", "pro")
    a.add_edge("A5", "C3", type="src")
    a.add_edu("E6", "So let us swim!")
    a.add_edge("E6", "A1", type="seg")
    return a


def get_very_complex_arggraph():
    a = ArgGraph(id="g1")
    a.add_edus(["txt1", "txt2", "txt3", "txt4"])
    a.add_edu_joint("j1", ["e1", "e2"])
    a.add_adu("a1", "pro")
    a.add_seg_edge("j1", "a1")
    a.add_edu_joint("j2", ["e3", "e4"])
    a.add_adu("a2", "opp")
    a.add_seg_edge("j2", "a2")
    a.add_edge_with_relation_node("c1", "a2", "a1", "reb")
    a.add_edu_adu("e5", "txt5", "a3", "opp")
    a.add_edge("a3", "c1", type="src")
    a.add_edu_adu("e6", "txt6", "a4", "pro")
    a.add_edge_with_relation_node("c2", "a4", "c1", "und")
    a.add_edu("e7", "txt7")
    a.add_seg_edge("e7", "a1")
    return a


def get_arggraph_restatement():
    a = ArgGraph(id="g1")
    a.add_edu_adu("e1", "Swim!", "a1", "pro")
    a.add_edu_adu("e2", "Good weather.", "a2", "pro")
    a.add_edge_with_relation_node("c1", "a2", "a1", "sup")
    a.add_edu("e3", "So swim!")
    a.add_seg_edge("e3", "a1")
    return a


def get_arggraph_joint():
    a = ArgGraph(id="g1")
    a.add_edus(["We.", "Should.", "Swim!"])
    a.add_edu_joint("j1", ["e1", "e2", "e3"])
    a.add_adu("a1", "pro")
    a.add_seg_edge("j1", "a1")
    a.add_edu_adu("e4", "Good weather.", "a2", "pro")
    a.add_edge_with_relation_node("c1", "a2", "a1", "sup")
    return a


def get_arggraph_linked():
    a = ArgGraph(id="g1")
    a.add_edu_adu("e1", "Swim!", "a1", "pro")
    a.add_edu_adu("e2", "Report says good weather", "a2", "pro")
    a.add_edu_adu("e3", "and report is always right.", "a3", "pro")
    a.add_edge_with_relation_node("c1", "a2", "a1", "sup")
    a.add_edge("a3", "c1", type="src")
    return a


def get_arggraph_center_embedding():
    a = ArgGraph(id="g1")
    a.add_edus(["We should,", "due to the good weather,", "go swimming!"])
    a.add_edu_joint("j1", ["e1", "e3"])
    a.add_adu("a1", "pro")
    a.add_seg_edge("j1", "a1")
    a.add_adu("a2", "pro")
    a.add_seg_edge("e2", "a2")
    a.add_edge_with_relation_node("c1", "a2", "a1", "sup")
    return a


def get_arggraph_flat_center_embedding():
    a = ArgGraph(id="g1")
    a.add_edus(["We should,", "due to the good weather,", "go swimming!"])
    a.add_edu_joint("j1", ["e1", "e2", "e3"])
    a.add_adu("a1", "pro")
    a.add_seg_edge("j1", "a1")
    a.add_adu("a2", "pro")
    a.add_seg_edge("e2", "a2")
    a.add_edge_with_relation_node("c1", "a2", "a1", "sup")
    return a


def get_arggraph_free_edu():
    g = ArgGraph(id="g")
    g.add_edus(["Well...", "Swim!", "Good weather."])
    g.add_adu("a1", "pro")
    g.add_seg_edge("e2", "a1")
    g.add_adu("a2", "pro")
    g.add_seg_edge("e3", "a2")
    g.add_edge_with_relation_node("c1", "a2", "a1", "sup")

    """
    >>> g.get_adu_segmented_text()
    ['Swim!', 'Good weather.']
    >>> g.get_segmented_text()
    ['Ok.', 'Swim!', 'Good weather.']
    >>> g.get_adu_segmented_text_with_restatements()
    ['Ok.', 'Swim!']
    >>> g.get_adu_adu_relations()
    [('a2', 'a1', 'sup')]
    >>> g.get_adus_as_dependencies()
    [('a2', 'a1', 'sup')]
    >>> g.get_edus_as_dependencies()
    [('e3', 'e2', 'sup')]
    >>> g.get_edus_as_dependencies(ids_to_numbers=True)
    [(3, 2, 'sup')]
    >>> g.get_edus_as_dependencies(ids_to_numbers=True, include_cc=True)
    [(1, 0, 'ROOT'), (3, 2, 'sup')]
    """
