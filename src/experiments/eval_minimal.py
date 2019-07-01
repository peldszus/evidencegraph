from argparse import ArgumentParser

from evidencegraph.argtree import SIMPLE_RELATION_SET
from evidencegraph.evaluation import evaluate_setting


if __name__ == '__main__':
    parser = ArgumentParser(
        description="""Evaluate argumentation parsing predictions""")
    parser.add_argument('--corpus', '-c', choices=CORPORA.keys(), default='m112en',
                        help='the corpus to evaluate the predictions of')
    args = parser.parse_args()
    corpus_name = args.corpus
    language = CORPORA[corpus_name]['language']

    settings = {
        ('adu', SIMPLE_RELATION_SET): [
            "{}-test-adu-simple-noop|equal".format(corpus_name)
        ]
    }

    for (segmentation, relationset), conditions in settings.iteritems():
        evaluate_setting(language, segmentation, relationset, conditions, corpus_id=corpus_name)
