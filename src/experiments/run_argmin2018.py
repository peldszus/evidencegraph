#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

'''
@author: Andreas Peldszus
'''


import json
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime

from numpy import mean

from evidencegraph.argtree import RELATION_SETS_BY_NAME
from evidencegraph.classifiers import EvidenceGraphClassifier
from evidencegraph.corpus import GraphCorpus, CORPORA
from evidencegraph.features_text import init_language
from evidencegraph.folds import get_static_folds
from evidencegraph.utils import hash_of_featureset

modelpath = "data/models/"


def experiment(in_corpus, out_corpus, folds, features, params, condition_name):
    maF1s = defaultdict(list)
    miF1s = defaultdict(list)
    predictions = defaultdict(dict)
    decisions = defaultdict(dict)

    for train_tids, test_tids, i in folds:
        print "[{}] Iteration: {}\t".format(datetime.now(), i)
        ensemble_basename = condition_name.split('|')[0]
        ensemble_name = "{}__{}__{}".format(
            ensemble_basename, hash_of_featureset(features.feature_set), i)
        clf = EvidenceGraphClassifier(
            features.feature_function_segments,
            features.feature_function_segmentpairs,
            **params
        )
        train_txt = [g for t, g in in_corpus.iteritems() if t in train_tids]
        train_arg = [g for t, g in out_corpus.iteritems() if t in train_tids]
        try:
            # load ensemble of pretrained base classifiers
            clf.load(modelpath + ensemble_name)
            if params['optimize_weighting']:
                # and train metaclassifier (if desired)
                clf.train_metaclassifier(train_txt, train_arg)
        except RuntimeError:
            # train ensemble
            clf.train(train_txt, train_arg)
            clf.save(modelpath + ensemble_name)

        # test
        test_txt = [g for t, g in in_corpus.iteritems() if t in test_tids]
        test_arg = [g for t, g in out_corpus.iteritems() if t in test_tids]
        score_msg = ''
        for level, base_classifier in clf.ensemble.items():
            maF1, miF1 = base_classifier.test(test_txt, test_arg)
            maF1s[level].append(maF1)
            miF1s[level].append(miF1)
            score_msg += "{}: {:.3f}\t".format(level, maF1)
        decoded_scores = []
        for t in test_tids:
            mst = clf.predict(in_corpus[t])
            decoded_scores.append(clf.score(mst, out_corpus[t]))
            predictions[i][t] = mst.get_triples()
            decisions[i][t] = clf.predict_decisions(in_corpus[t])
        score_msg += "decoded: {:.3f}\t".format(mean(decoded_scores))
        print score_msg

    print "Average macro and micro F1:"
    for level in maF1s:
        avg_maF1 = mean(maF1s[level])
        avg_miF1 = mean(miF1s[level])
        print level, avg_maF1, avg_miF1

    return predictions, decisions


def create_corpus(corpora, mode="normal"):
    """
    corpora: list of corpus ids to load
    mode:
        'normal' = train and test folded on all corpora
                   A" + B" + C" >> A' + B' + C'
        'cross'  = train on all but last corpus, test folded on last
                   A + B >> C'
        'add'    = train on all and last corpus, test folded on last
                   A + B + C" >> C'
    """
    assert all(corpus in CORPORA for corpus in corpora)

    if mode == "normal":
        gc = GraphCorpus()
        for corpus in corpora:
            gc.load(CORPORA[corpus]['path'])
        folds = list(gc.create_folds())

    elif mode == "cross":
        gc = GraphCorpus()
        assert len(corpora) > 1
        last = corpora[-1]
        gc.load(CORPORA[last]['path'])
        last_folds = list(gc.create_folds())
        first_corpora_text_ids = []
        for corpus in corpora[:-1]:
            ids = gc.load(CORPORA[corpus]['path'])
            first_corpora_text_ids.extend(ids)
        folds = [(first_corpora_text_ids, test, n)
                 for _, test, n in last_folds]

    elif mode == "add":
        gc = GraphCorpus()
        assert len(corpora) > 1
        last = corpora[-1]
        all_corpora_text_ids = []
        ids = gc.load(CORPORA[last]['path'])
        all_corpora_text_ids.extend(ids)
        last_folds = list(gc.create_folds())
        for corpus in corpora[:-1]:
            ids = gc.load(CORPORA[corpus]['path'])
            all_corpora_text_ids.extend(ids)
        folds = [([i for i in all_corpora_text_ids if i not in test], test, n)
                 for _, test, n in last_folds]

    return gc, folds


if __name__ == '__main__':
    language = "en"

    # features sets
    features_old = [
        'default', 'bow', 'bow_2gram', 'first_three',
        'tags', 'deps_lemma', 'deps_tag',
        'punct', 'verb_main', 'verb_all', 'discourse_marker',
        'context'
    ]
    features_new_non_vector = [
        'clusters', 'clusters_2gram', 'discourse_relation',
        'vector_left_right', 'vector_source_target',
        'verb_segment', 'same_sentence', 'matrix_clause'
    ]
    features_all_but_vector = features_old + features_new_non_vector

    conditions = {
        # evaluate on old corpus (m112en)
        'argmin2018-normal-m112en-adu-simple-op|equal': {
            'corpora': ['m112en'],
            'mode': 'normal',
            'feature_set': features_all_but_vector,
            'relation_set': "SIMPLE_RELATION_SET",
            'segmentation': 'adu',
            'optimize': True,
            'optimize_weighting': False
        },
        'argmin2018-cross-part2+m112en-adu-simple-op|equal': {
            'corpora': ['m112en_part2', 'm112en'],
            'mode': 'cross',
            'feature_set': features_all_but_vector,
            'relation_set': "SIMPLE_RELATION_SET",
            'segmentation': 'adu',
            'optimize': True,
            'optimize_weighting': False
        },
        'argmin2018-add-part2+m112en-adu-simple-op|equal': {
            'corpora': ['m112en_part2', 'm112en'],
            'mode': 'add',
            'feature_set': features_all_but_vector,
            'relation_set': "SIMPLE_RELATION_SET",
            'segmentation': 'adu',
            'optimize': True,
            'optimize_weighting': False
        },

        # evaluate on new corpus (m112en_part2)
        'argmin2018-normal-part2-adu-simple-op|equal': {
            'corpora': ['m112en_part2'],
            'mode': 'normal',
            'feature_set': features_all_but_vector,
            'relation_set': "SIMPLE_RELATION_SET",
            'segmentation': 'adu',
            'optimize': True,
            'optimize_weighting': False
        },
        'argmin2018-cross-m112en+part2-adu-simple-op|equal': {
            'corpora': ['m112en', 'm112en_part2'],
            'mode': 'cross',
            'feature_set': features_all_but_vector,
            'relation_set': "SIMPLE_RELATION_SET",
            'segmentation': 'adu',
            'optimize': True,
            'optimize_weighting': False
        },
        'argmin2018-add-m112en+part2-adu-simple-op|equal': {
            'corpora': ['m112en', 'm112en_part2'],
            'mode': 'add',
            'feature_set': features_all_but_vector,
            'relation_set': "SIMPLE_RELATION_SET",
            'segmentation': 'adu',
            'optimize': True,
            'optimize_weighting': False
        }
    }

    # run all experiment conditions
    for condition_name, params in conditions.items():
        print "### Running experiment condition", condition_name

        # load and combine corpora
        corpus, folds = create_corpus(
            params.pop('corpora'), mode=params.pop('mode'))

        # set condition params
        features = init_language(language)
        features.feature_set = params.pop('feature_set')
        params['relation_set'] = RELATION_SETS_BY_NAME.get(
            params['relation_set'])
        texts, trees = corpus.segments_trees(
            params.pop('segmentation'), params['relation_set'])

        # health check
        for tid in sorted(trees):
            triples = trees[tid].get_triples(include_root=True)
            assert len(triples) == len(texts[tid]), "Somethings wrong with {}".format(tid)

        # run condition
        predictions, _decisions = experiment(
            texts, trees, folds, features, params, condition_name)
        with open('data/{}.json'.format(condition_name), 'w') as f:
            json.dump(predictions, f, indent=1, sort_keys=True)
