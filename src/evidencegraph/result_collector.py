#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
@author: Andreas Peldszus
'''

from __future__ import print_function

from time import strftime
import gzip
import cPickle as pickle

import pandas as pd
from scipy.stats import wilcoxon
from sklearn.base import BaseEstimator


def is_numeric(obj):
    """
    Returns true if `obj` is a numerical object.

    Args:
        obj (object): the object to check

    Returns:
        bool: True if the object is numerical, otherwise False.

    >>> is_numeric(100)
    True
    >>> is_numeric(0.5)
    True
    >>> is_numeric("hallo")
    False
    >>> is_numeric({1: 'a', 2: 'b'})
    False
    >>> is_numeric([1, 2, 3, 4])
    False
    >>> is_numeric({1, 2, 3, 4})
    False
    """
    # http://stackoverflow.com/a/500908
    attrs = ['__add__', '__sub__', '__mul__', '__div__', '__pow__']
    return all(hasattr(obj, attr) for attr in attrs)


def value_in_nested_dict(d, path_of_keys):
    """
    Follow a path of keys in a nested dict and return the value found
    at the end of the path.

    Args:
        d (dict): a (nested) dictionary with values
        path_of_keys (list): a list of keys

    Returns:
        object, if a value is found at the end of the path
        None, otherwise.

    >>> d = {'a': {1: 0.5, 2: 0.3}, 'b': 0.8}
    >>> value_in_nested_dict(d, ['a', 1])
    0.5
    >>> value_in_nested_dict(d, ['b'])
    0.8

    If `path_of_keys` contains keys not found in `d`, return None
    >>> value_in_nested_dict(d, ['X', 0.5]) is None
    True
    """
    if len(path_of_keys) == 0:
        return d
    else:
        try:
            return value_in_nested_dict(d[path_of_keys[0]], path_of_keys[1:])
        except KeyError:
            return None


def load_result_collector(filename):
    """
    Loads a dumped collected results from a gzip pickle file and returns it.

    Args:
        filename (str): path to the file to load

    Returns:
        ResultCollector: the loaded result collectors
    """
    with gzip.open(filename, 'rb') as f:
        return pickle.load(f)


class ResultCollector(object):
    """
    A datastructure for collecting the results of an experiment with one or
    condition, iterations, levels and metrics.

    For a specific combination of condition, iteration and level, experimental
    results results can be added in form of a dictionary, which contains the
    scores of different metrics.
    """

    def __init__(self, name='noname', series='noseries', desc='nodesc'):
        # meta data
        self.name = name
        self.series = series
        self.desc = desc
        self.timestamp = strftime("%Y-%m-%d_%X")
        # registers
        self.conditions = list()
        self.levels = list()
        self.iterations = list()
        # data store
        self.data = list()
        self.path_to_metric = list()

    def add_result(self, condition, iteration, level, data):
        """
        Adds a result data point for specific condition, iteration and level to
        the datastructure.

        If the condition, iteration or level isn't registered yet, it is added
        to the experiment register.

        Args:
            condition (str): an identifier of the experiment condition
            iteration (str): an identifier of the experiment iteration
            level (str): an identifier of the experiment level
            data (dict): a (nested) dictionary mapping metric names to values

        >>> rc = ResultCollector()
        >>> rc.add_result('condition1', 'iteration2', 'level3', {'score': 1.0})
        >>> 'condition1' in rc.conditions
        True
        >>> 'iteration2' in rc.iterations
        True
        >>> 'level3' in rc.levels
        True
        >>> rc.data
        [('condition1', 'iteration2', 'level3', {'score': 1.0})]
        """
        if condition not in self.conditions:
            self.conditions.append(condition)
        if iteration not in self.iterations:
            self.iterations.append(iteration)
        if level not in self.levels:
            self.levels.append(level)
        # TODO: check for double entries
        self.data.append((condition, iteration, level, data))

    def set_metric(self, path_of_keys, ignore_type=False):
        """
        Define the metric to be considered from the data points.

        Args:
            path_of_keys (list): a list of keys for accessing the (nested)
                data dicts
            ignore_type (bool): ignore or to check whether `path_of_keys` does
                lead to a numerical value

        Raises:
            ValueError: if `path_of_keys` does not lead to a numerical value

        >>> rc = ResultCollector()
        >>> rc.add_result('c', 1, 'l', {'score': 1.0, 'ids': [1, 2, 3]})
        >>> rc.path_to_metric
        []

        >>> rc.set_metric(['score'])
        >>> rc.path_to_metric
        ['score']
        >>> rc.set_metric(['ids'])
        Traceback (most recent call last):
            ...
        ValueError: The path of keys does not lead to anumerical object/number.

        >>> rc.set_metric(['ids'], ignore_type=True)
        >>> rc.path_to_metric
        ['ids']
        """
        # test path first
        if len(self.data) > 0:
            v = value_in_nested_dict(self.data[0][3], path_of_keys)
            if (not ignore_type) and (not is_numeric(v)):
                raise ValueError(('The path of keys does not lead to a'
                                  'numerical object/number.'))
        else:
            print("Warning: path_of_keys to the metric cannot be validated.")
        self.path_to_metric = path_of_keys

    def save(self, filename):
        """
        Saves the collected results as a gzip pickle file.

        Args:
            filename (str): path to the file to write to
        """
        with gzip.open(filename, 'wb') as f:
            pickle.dump(self, f, -1)

    def print_result(self, condition, level):
        """
        Prints a summary of the results over all iterations
        on a certain level under a certain condition

        Args:
            condition (str): an identifier of the experiment condition
            level (str): an identifier of the experiment level

        >>> rc = ResultCollector()
        >>> rc.add_result('c', 1, 'l', {'score': 1.00})
        >>> rc.add_result('c', 2, 'l', {'score': 1.50})
        >>> rc.add_result('c', 3, 'l', {'score': 0.25})
        >>> rc.set_metric(['score'])
        >>> rc.print_result('c', 'l')
        count    3.000000
        mean     0.916667
        std      0.629153
        min      0.250000
        25%      0.625000
        50%      1.000000
        75%      1.250000
        max      1.500000
        dtype: float64
        """
        print(self._sum_result(condition, level))

    def print_result_for_level(self, level, print_header=True):
        """
        Prints a summary of the results over all iterations
        under all conditions on a certain level

        Args:
            level (str): an identifier of the experiment level

        >>> rc = ResultCollector()
        >>> rc.add_result('c1', 1, 'l1', {'score': .30})
        >>> rc.add_result('c1', 2, 'l1', {'score': .50})
        >>> rc.add_result('c2', 1, 'l1', {'score': .25})
        >>> rc.add_result('c2', 2, 'l1', {'score': .40})
        >>> rc.set_metric(['score'])
        >>> rc.print_result_for_level('l1')
        level  c1  c2
        l1  0.400 (+- 0.141)    0.325 (+- 0.106)
        """
        if print_header:
            print('\t'.join(['level'] + self.conditions))
        print('\t'.join([level] + [self._string_summary(condition, level)
                                   for condition in self.conditions]))

    def print_all_results(self):
        """
        Prints a summary of the all results over all conditions and levels.

        >>> rc = ResultCollector()
        >>> rc.add_result('c1', 1, 'l1', {'score': .30})
        >>> rc.add_result('c1', 2, 'l1', {'score': .50})
        >>> rc.add_result('c1', 1, 'l2', {'score': .25})
        >>> rc.add_result('c1', 2, 'l2', {'score': .50})
        >>> rc.add_result('c2', 1, 'l1', {'score': .25})
        >>> rc.add_result('c2', 2, 'l1', {'score': .40})
        >>> rc.add_result('c2', 1, 'l2', {'score': .45})
        >>> rc.add_result('c2', 2, 'l2', {'score': .35})
        >>> rc.set_metric(['score'])
        >>> rc.print_all_results()
        level  c1  c2
        l1  0.400 (+- 0.141)    0.325 (+- 0.106)
        l2  0.375 (+- 0.177)    0.400 (+- 0.071)
        """
        print('\t'.join(['level'] + self.conditions))
        for level in self.levels:
            print('\t'.join([level] + [self._string_summary(condition, level)
                                       for condition in self.conditions]))

    def wilcoxon(self, conditionA, conditionB, level):
        """
        Calculate the Wilcoxon signed-rank test for comparing two conditions
        on a certain level. See scipy.stats.wilcoxon.__doc__

        Args:
            conditionA (string): an identifier of one experiment condition
            conditionB (string): an identifier of another experiment condition
            level (string): an identifier of the experiment level

        Returns:
            statistic (float): The sum of the ranks of the differences above
                or below zero, whichever is smaller.
            pvalue (float): The two-sided p-value for the test.

        >>> rc = ResultCollector()
        >>> rc.add_result('c1', 1, 'l1', {'score': .30})
        >>> rc.add_result('c1', 2, 'l1', {'score': .50})
        >>> rc.add_result('c2', 1, 'l1', {'score': .25})
        >>> rc.add_result('c2', 2, 'l1', {'score': .40})
        >>> rc.set_metric(['score'])
        >>> rc.wilcoxon('c1', 'c2', 'l1')
        (0.0, 0.17971249487899976)
        """
        result_a = self._get_result(conditionA, level)
        result_b = self._get_result(conditionB, level)
        if result_a == result_b:
            statistic, pvalue = None, 1.0
        else:
            statistic, pvalue = wilcoxon(result_a, result_b)
        return statistic, pvalue

    def _get_result(self, condition, level):
        assert condition in self.conditions
        assert level in self.levels
        relevant_data = [value_in_nested_dict(d, self.path_to_metric)
                         for c, _i, l, d in self.data
                         if c == condition and l == level]
        return relevant_data

    def _sum_result(self, condition, level):
        relevant_data = self._get_result(condition, level)
        return pd.Series(relevant_data).describe()

    def _string_summary(self, condition, level):
        t = self._sum_result(condition, level)
        return "%.3f (+- %.3f)" % (t['mean'], t['std'])


def filter_params(params):
    ''' this function can be used to filter the get_params output for
        Estimator instances, so that no objects but only their string
        representations are pickled '''
    out = {}
    for k, v in params.iteritems():
        if isinstance(v, dict):
            v2 = filter_params(v)
        elif isinstance(v, BaseEstimator) or hasattr(v, '__call__'):
            v2 = str(v).replace('\n      ', '')
        else:
            v2 = v
        out[k] = v2
    return out
