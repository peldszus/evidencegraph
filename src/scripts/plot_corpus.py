"""
Created on 05.08.2014

@author: andi
"""


import argparse
import sys

from evidencegraph.arggraph import ArgGraph


if __name__ == "__main__":
    aparser = argparse.ArgumentParser(
        description="plot the arg graph specified in the input xml as a png file"
    )
    aparser.add_argument("input", help="input xml file(s)", nargs="+")
    args = aparser.parse_args(sys.argv[1:])

    for i in args.input:
        if i.endswith(".xml"):
            print(i, "...")
            id = i[:-4]
            g = ArgGraph()
            g.load_from_xml(i)

            # export dot, png and pdf of graph
            # with open(id + '.dot', 'w') as f:
            #     f.write(g.render_as_dot())
            g.render_as_png(id + ".png")
            g.render_as_pdf(id + ".pdf")

            # new_xml = g.to_xml()
            # with open(i[:-4]+'.new.xml', 'w') as f:
            #     f.write(new_xml)

            # # export dot, png and pdf of reduced graph
            # r = g.get_relation_node_free_graph()
            # with open(id + '_reduced.dot', 'w') as f:
            #     f.write(r.render_as_dot())
            # r.render_as_png(id + '_reduced.png')
            # r.render_as_pdf(id + '_reduced.pdf')

            # # export segments and text
            # segs = g.get_segmented_text()
            # with codecs.open(id + '.adus', 'w', 'utf8') as f:
            #     f.write('\n'.join(segs))
            # text = g.get_unsegmented_text()
            # with codecs.open(id + '.txt', 'w', 'utf8') as f:
            #     f.write(text)
