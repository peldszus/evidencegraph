from argparse import ArgumentParser

from evidencegraph.argtree import SIMPLE_RELATION_SET
from evidencegraph.evaluation import evaluate_setting


if __name__ == '__main__':
    parser = ArgumentParser(
        description="""Evaluate argumentation parsing predictions""")
    parser.add_argument('--lang', '-l', choices=['en', 'de'], default='en',
                        help='the language to consider the predictions of')
    args = parser.parse_args()
    language = args.lang
    corpus_name = 'm112{}'.format(language)

    settings = {
        ('adu', SIMPLE_RELATION_SET): [
            "{}-test-adu-simple-noop|equal".format(corpus_name)
        ]
    }

    for (segmentation, relationset), conditions in settings.iteritems():
        evaluate_setting(language, segmentation, relationset, conditions, corpus_id=corpus_name)
