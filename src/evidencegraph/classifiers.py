#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

"""
Created on 06.05.2016

@author: Andreas Peldszus
"""
from __future__ import print_function
from __future__ import absolute_import

import os
from functools import partial

import joblib
from numpy import mean, zeros
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_fscore_support

from .argtree import FULL_RELATION_SET
from .decode import find_mst as find_mst
from .result_collector import filter_params
from .evidence_graph import EvidenceGraph
from .search import EvolutionarySearch
from .features_text import generate_items_segments
from .features_text import generate_items_segmentpairs
from .utils import foldsof


def label_function_cc(argtree):
    return argtree.get_cc_vector()


def label_function_ro(argtree):
    return argtree.get_ro_vector()


def label_function_fu(argtree):
    return argtree.get_fu_vector()


def label_function_at(argtree):
    return argtree.get_at_vector()


class BaseClassifier(object):

    DEFAULT_PARAM_SEARCH = {
        "sgd__alpha": [0.001, 0.005],
        "sgd__n_iter": [100, 250, 500],
        "sgd__l1_ratio": [0.1, 0.15, 0.2, 0.25],
        # 'sgd__average': [False, 50, 100, 500, 1],
        # 'k_b__k': [50, 100, 500, 'all'],
    }

    DEFAULT_PARAMS_FIXED = {
        "k_b__k": "all",
        "sgd__loss": "log",
        "sgd__penalty": "elasticnet",
        "sgd__learning_rate": "optimal",
        "sgd__class_weight": "auto",
        "sgd__alpha": 0.005,
        "sgd__n_iter": 500,
        "sgd__l1_ratio": 0.2,
        "sgd__n_jobs": -1,
    }

    DEFAULT_PIPELINE = [
        ("vec", DictVectorizer),
        ("var", VarianceThreshold),
        ("k_b", SelectKBest),
        ("sgd", SGDClassifier),
    ]

    def __init__(self, feature_function=None, label_function=None):
        # instantiate all estimators in the pipeline
        self.pipeline = Pipeline(
            [
                (name, estimator())
                for name, estimator in BaseClassifier.DEFAULT_PIPELINE
            ]
        )
        self.pipeline.set_params(**BaseClassifier.DEFAULT_PARAMS_FIXED)
        self.feature_function = feature_function
        self.label_function = label_function

    def train(self, in_data, gold_data):
        assert len(in_data) == len(gold_data)
        features = [
            feature
            for datum in in_data
            for feature in self.extract_features(datum)
        ]
        labels = [
            label
            for datum in gold_data
            for label in self.extract_labels(datum)
        ]
        self._train(features, labels)

    def train_optimize(self, in_data, gold_data, verbose=True):
        self.pipeline = GridSearchCV(
            self.pipeline,
            BaseClassifier.DEFAULT_PARAM_SEARCH,
            n_jobs=-1,
            cv=3,
            scoring="log_loss",
        )
        self.train(in_data, gold_data)
        self.best_params = filter_params(
            self.pipeline.best_estimator_.get_params(deep=False)
        )
        if verbose:
            print (self.best_params)

    def _train(self, features, labels):
        self.pipeline.fit(features, labels)

    def predict_collection(self, in_data, prediction_type="class"):
        features = [
            feature
            for datum in in_data
            for feature in self.extract_features(datum)
        ]
        return self._predict(features, prediction_type=prediction_type)

    def predict(self, in_datum, prediction_type="class"):
        features = self.extract_features(in_datum)
        return self._predict(features, prediction_type=prediction_type)

    def _predict(self, features, prediction_type="class"):
        if prediction_type == "proba":
            return self.pipeline.predict_proba(features)
        elif prediction_type == "decision":
            return self.pipeline.decision_function(features)
        else:
            return self.pipeline.predict(features)

    def test(self, in_data, gold_data):
        predicted_labels = self.predict_collection(in_data)
        labels = [
            label
            for datum in gold_data
            for label in self.extract_labels(datum)
        ]
        return self._score(labels, predicted_labels)

    def _score(self, gold, pred):
        _, _, macro_f1, _ = precision_recall_fscore_support(
            gold, pred, average="macro", pos_label=None, warn_for=()
        )
        _, _, micro_f1, _ = precision_recall_fscore_support(
            gold, pred, average="micro", pos_label=None, warn_for=()
        )
        return macro_f1, micro_f1

    def extract_features(self, in_datum):
        return self.feature_function(in_datum)

    def extract_labels(self, gold_datum):
        return self.label_function(gold_datum)


class EvidenceGraphClassifier(object):
    def __init__(
        self,
        feature_function_segments,
        feature_function_segmentpairs,
        optimize=True,
        optimize_weighting=False,
        relation_set=FULL_RELATION_SET,
        base_classifier_class=BaseClassifier,
    ):
        self.feature_function_segments = feature_function_segments
        self.feature_function_segmentpairs = feature_function_segmentpairs
        self.relation_set = relation_set
        self.base_classifier_class = base_classifier_class
        self.ensemble = {
            "cc": base_classifier_class(
                feature_function=feature_function_segments,
                label_function=label_function_cc,
            ),
            "ro": base_classifier_class(
                feature_function=feature_function_segments,
                label_function=label_function_ro,
            ),
            "fu": base_classifier_class(
                feature_function=feature_function_segments,
                label_function=label_function_fu,
            ),
            "at": base_classifier_class(
                feature_function=feature_function_segmentpairs,
                label_function=label_function_at,
            ),
        }
        self.optimize = optimize
        self.optimize_weighting = optimize_weighting
        self.weighting = {level: 0.25 for level in self.ensemble.keys()}

    def train(self, input_trees, output_trees):
        # train base classifiers
        for level, clf in self.ensemble.iteritems():
            if self.optimize:
                clf.train_optimize(input_trees, output_trees, verbose=False)
            else:
                clf.train(input_trees, output_trees)
        # train meta classifier
        if self.optimize_weighting:
            self.train_metaclassifier(input_trees, output_trees)

    def train_metaclassifier(self, input_trees, output_trees):
        if self.optimize_weighting == "inner_cv":
            # predict all items in trainingset as unseen via inner CV
            egs = []
            for (train_X, train_y), (test_X, test_y) in foldsof(
                input_trees, output_trees
            ):
                egclf = EvidenceGraphClassifier(
                    self.feature_function_segments,
                    self.feature_function_segmentpairs,
                    relation_set=self.relation_set,
                    base_classifier_class=self.base_classifier_class,
                    optimize=False,
                    optimize_weighting=False,
                )
                egclf.train(train_X, train_y)
                fold_egs = [egclf._predict_evidence_graph(t) for t in test_X]
                del egclf
                egs.extend(fold_egs)
        else:
            egs = [self._predict_evidence_graph(t) for t in input_trees]

        def weighting_dict(w1, w2, w3, w4):
            return {"cc": w1, "ro": w2, "fu": w3, "at": w4}

        def score_weighting(w1, w2, w3, w4, items=None):
            weighting = weighting_dict(w1, w2, w3, w4)
            scores = []
            for eg, gold in items:
                mst = self._decode(eg, weighting=weighting)
                scores.append(self.score(mst, gold))
            return mean(scores)

        callback = partial(score_weighting, items=zip(egs, output_trees))
        search = EvolutionarySearch(callback, n_to_start_with=20)
        search.search(verbose=True)
        search.report()
        self.weighting = weighting_dict(*search.get_best())

    def predict_collection(self, input_trees):
        for tree in input_trees:
            yield self.predict(tree)

    def _extend_probas_to_all_classes(self, probas):
        # if there are fu classes in the test set which do not occurred in the
        # training set, then the vector returned by predict_proba will only cover
        # those found in the training. We need to extend it to the full set of
        # classes in order index it via class columns.
        fu_classes_all = sorted(
            self.relation_set.map_function_to_vector.values()
        )
        clf = self.ensemble["fu"].pipeline
        if isinstance(clf, GridSearchCV):
            fu_classes_learned = clf.best_estimator_.steps[-1][1].classes_
        elif isinstance(clf, Pipeline):
            fu_classes_learned = clf.steps[-1][1].classes_
        else:
            raise Exception("Unknown classifier object")
        result_shape = (len(probas), len(fu_classes_all))
        result = zeros(result_shape)
        for idx, cl in enumerate(fu_classes_learned):
            result[:, cl] = probas[:, idx]
        return result

    def _predict(self, input_tree, prediction_type="proba"):
        # predict with base classifiers
        predictions = {}
        for level, clf in self.ensemble.iteritems():
            predictions[level] = clf.predict(
                input_tree, prediction_type=prediction_type
            )

        # extend fu proba vector dimensions to all existing classes
        if prediction_type == "proba":
            predictions["fu"] = self._extend_probas_to_all_classes(
                predictions["fu"]
            )

        # reconstruct item identity from vector column
        items_segments = generate_items_segments(input_tree)
        items_pairs = generate_items_segmentpairs(input_tree)

        def itemize(level, preds):
            if level == "at":
                return dict(zip(items_pairs, preds))
            else:
                return dict(zip(items_segments, preds))

        itemized_predictions = {
            level: itemize(level, preds)
            for level, preds in predictions.iteritems()
        }
        return itemized_predictions

    def predict_decisions(self, input_tree):
        return self._predict(input_tree, prediction_type="decision")

    def predict(self, input_tree):
        eg = self._predict_evidence_graph(input_tree)
        mst = self._decode(eg)
        return mst

    def _predict_evidence_graph(self, input_tree):
        itemized_predictions = self._predict(
            input_tree, prediction_type="proba"
        )
        eg = self._build_evidence_graph(itemized_predictions)
        return eg

    def _decode(self, evidence_graph, weighting=None):
        if weighting is None:
            weighting = self.weighting
        weg = evidence_graph.get_weighted_evidence_graph(weights=weighting)
        mst = find_mst(weg)
        mst.relation_set = self.relation_set
        return mst

    def score(self, input_tree, output_tree):
        pred = input_tree.get_vector()
        gold = output_tree.get_vector()
        score = 1.0
        for level in self.ensemble.keys():
            _, _, macro_f1, _ = precision_recall_fscore_support(
                gold[level],
                pred[level],
                average="macro",
                pos_label=None,
                warn_for=(),
            )
            score *= macro_f1
        return score

    def _build_evidence_graph(self, itemized_predictions):
        eg = EvidenceGraph(weight_ids=["cc", "ro", "fu", "at"])
        map_fu_to_vec = self.relation_set.map_function_to_vector
        for (source, target), p_at in itemized_predictions["at"].iteritems():
            p_cc = itemized_predictions["cc"][source]
            p_ro_source = itemized_predictions["ro"][source]
            p_ro_target = itemized_predictions["ro"][target]
            p_fu = itemized_predictions["fu"][source]
            for func_type in map_fu_to_vec.keys():
                if func_type in ["cc", "unknown"]:
                    continue
                # probability of attachment
                at_weight = p_at[1]
                # probability of not being the central claim
                cc_weight = p_cc[0]
                # probability of role switch
                if func_type in self.relation_set.functions_inverting_role:
                    ro_weight = (
                        p_ro_source[0] * p_ro_target[1]
                        + p_ro_source[1] * p_ro_target[0]
                    )
                else:
                    ro_weight = (
                        p_ro_source[0] * p_ro_target[0]
                        + p_ro_source[1] * p_ro_target[1]
                    )
                # probability of edge type
                try:
                    fu_weight = p_fu[map_fu_to_vec[func_type]]
                except IndexError:
                    print (p_fu)
                    print (map_fu_to_vec)
                    print (func_type)
                eg.add_edge(
                    source,
                    target,
                    type=func_type,
                    cc=cc_weight,
                    ro=ro_weight,
                    fu=fu_weight,
                    at=at_weight,
                )
        return eg

    def save(self, path, verbose=True):
        """Save ensemble of base classifiers."""
        pipelines = [
            (lvl, self.ensemble[lvl].pipeline)
            for lvl in sorted(self.ensemble.keys())
        ]
        joblib.dump(pipelines, path, compress=1)
        if verbose:
            print ("Saved model {}".format(path))

    def load(self, path, verbose=True):
        """Load an object (typically a classifier model) using joblib."""
        if not os.path.isfile(path) or not os.access(path, os.R_OK):
            raise RuntimeError("Can't load model from file {:s}".format(path))
        pipelines = joblib.load(path)
        for lvl, pipeline in pipelines:
            self.ensemble[lvl].pipeline = pipeline
        if verbose:
            print ("Loaded model {}".format(path))
