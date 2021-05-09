"""
@author: Andreas Peldszus
"""


import sys

import numpy as np


class BasicWeightingSearch:
    def __init__(self, calc_function):
        """
        An abstract search for the best weighting, represented as a
        vector of four positive floats summing up to 1.
        The search-function of this class should be implemented.
        """
        self.calc_func = calc_function
        self.reset()

    def reset(self):
        """
        Forget about tested weightings.

        >>> s = BasicWeightingSearch(lambda *x: 1.0)
        >>> s.scores = {(0.5, 0.3, 0.2, 0.0): 1.0}
        >>> s.reset()
        >>> s.scores
        {}
        """
        self.scores = {}

    def search(self):
        raise NotImplementedError

    def test_weighting(self, w1, w2, w3, w4):
        """
        Tests a given weighting by applying the score function
        and saving the resulting score.

        >>> s = BasicWeightingSearch(lambda *x: 1.0)
        >>> w = (0.5, 0.3, 0.2, 0.0)
        >>> s.test_weighting(*w)
        >>> s.scores
        {(0.5, 0.3, 0.2, 0.0): 1.0}
        """
        score = self.calc_func(w1, w2, w3, w4)
        self.scores[(w1, w2, w3, w4)] = score

    def _random_weighting(self):
        """
        Generates some random weighing: four floats summing up to 1.

        >>> s = BasicWeightingSearch(lambda *x: 1.0)
        >>> r = s._random_weighting()
        >>> abs(1 - sum(r)) < 1e-6
        True
        """
        x = np.random.random(4)
        return tuple(x / sum(x))

    def get_best(self):
        """
        Returns the highest scoring weighting saved so far.

        >>> s = BasicWeightingSearch(lambda *x: 1.0)
        >>> s.scores = {(0.5, 0.3, 0.2, 0.0): 1.0, (0.0, 0.2, 0.3, 0.5): 0.5}
        >>> s.get_best()
        (0.5, 0.3, 0.2, 0.0)
        """
        return max(self.scores, key=self.scores.get)

    def report(self):
        """
        Prints a short report about the variation of the scored weightings.
        """
        n = len(self.scores)
        min_score = min(self.scores.values())
        max_score = max(self.scores.values())
        print(
            "Searched {} weightings, scoring from {} up to {}.".format(
                n, min_score, max_score
            )
        )


class ThrowRiceSearch(BasicWeightingSearch):
    def __init__(self, calc_function, n=99):
        """
        A very simple search, which draws `n` random
        weightings and tests the score of all of them.

        >>> from scipy.spatial.distance import cosine
        >>> f = lambda *w: 1.0 / cosine(w, [0.5, 0.3, 0.2, 0.0])
        >>> s = ThrowRiceSearch(f, n=1000)
        >>> s.search()
        >>> w = s.get_best()
        >>> w[0] > w[1] > w[2] > w[3]
        True
        """
        super().__init__(calc_function)
        self.n = n

    def search(self, verbose=False):
        weightings_to_search_in = [(0.25, 0.25, 0.25, 0.25)] + [
            self._random_weighting() for _ in range(self.n)
        ]
        searched_weightings = 0
        for weighting in weightings_to_search_in:
            if searched_weightings > 0 and searched_weightings % 100 == 0:
                if verbose:
                    sys.stdout.write("\n")
            if verbose:
                sys.stdout.write(".")
            searched_weightings += 1
            self.test_weighting(*weighting)
        if verbose:
            print("!")


class EvolutionarySearch(BasicWeightingSearch):
    def __init__(
        self,
        calc_function,
        n_to_start_with=20,
        n_to_keep_proportion=0.25,
        factor=0.5,
        stop_after=1e-4,
        stop_after_step=3,
    ):
        """
        A simple evolutionary search, which starts with some randomly
        drawn weightings, and then gradually refines the best weightings
        by randomly jittering them with a decreasing jitter rate.

        >>> from scipy.spatial.distance import cosine
        >>> f = lambda *w: 1.0 / cosine(w, [0.5, 0.3, 0.2, 0.0])
        >>> s = EvolutionarySearch(f)
        >>> s.search()
        >>> w = s.get_best()
        >>> w[0] > w[1] > w[2] > w[3]
        True
        """
        super().__init__(calc_function)
        self.n_to_start_with = n_to_start_with
        self.n_to_keep_proportion = n_to_keep_proportion
        self.factor = factor
        self.stop_after = stop_after
        self.stop_after_step = stop_after_step

    def search(self, verbose=False):
        n_to_keep = int(self.n_to_start_with * self.n_to_keep_proportion)
        # start by scoring the first weightings
        weightings_to_start_with = [(0.25, 0.25, 0.25, 0.25)] + [
            self._random_weighting() for _ in range(self.n_to_start_with)
        ]
        for weighting in weightings_to_start_with:
            self.test_weighting(*weighting)
        # then get top n_to_keep_proportion weighings and jitter them
        last_top_score = 0.0
        rate = 1.0
        step = 0
        while True:
            top_weightings = sorted(
                [
                    (score, weighting)
                    for weighting, score in self.scores.items()
                ],
                reverse=True,
            )
            # stoping criterion
            top_score = top_weightings[0][0]
            if (top_score - last_top_score) < self.stop_after:
                if step == self.stop_after_step:
                    break
                else:
                    step += 1
            else:
                last_top_score = top_score
                step = 0
            rate = rate * self.factor
            if verbose:
                best = top_weightings[0]
                print(
                    "### rate=%.4f - top weighting: score=%.3f weighting=(%s)"
                    % (rate, best[0], ", ".join(["%.3f" % w for w in best[1]]))
                )
            # jitter!
            for _score, weighting in top_weightings[:n_to_keep]:
                for _i in range(int(self.n_to_start_with / n_to_keep / 4)):
                    for jittered in self._jitter_weighting(weighting, rate):
                        self.test_weighting(*jittered)

    def _jitter_weighting(self, w, r):
        for i in range(len(w)):
            x = np.array(w)
            x[i] += r
            yield tuple(x / sum(x))
