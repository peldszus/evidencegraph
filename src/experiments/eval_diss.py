from evidencegraph.argtree import SIMPLE_RELATION_SET
from evidencegraph.argtree import FULL_RELATION_SET
from evidencegraph.argtree import FULL_RELATION_SET_ADU
from evidencegraph.evaluation import evaluate_setting


if __name__ == "__main__":
    settings = {
        # de adu
        ("adu", SIMPLE_RELATION_SET, "de", "m112de"): [
            "m112{}-diss-adu-simple-baseline-first",
            "m112{}-diss-adu-simple-baseline-prec",
            # 'm112{}-emnlp2015-predictions',
            "m112{}-diss-adu-simple-op|equal",
            "m112{}-diss-adu-simple-op|train",
            "m112{}-diss-adu-simple-op|test",
        ],
        ("adu", FULL_RELATION_SET_ADU, "de", "m112de"): [
            "m112{}-diss-adu-full-baseline-first",
            "m112{}-diss-adu-full-baseline-prec",
            "m112{}-diss-adu-full-op|equal",
            "m112{}-diss-adu-full-op|train",
            "m112{}-diss-adu-full-op|test",
        ],
        # en adu
        ("adu", SIMPLE_RELATION_SET, "en", "m112en"): [
            "m112{}-diss-adu-simple-baseline-first",
            "m112{}-diss-adu-simple-baseline-prec",
            # 'm112{}-emnlp2015-predictions',
            "m112{}-diss-adu-simple-op|equal",
            "m112{}-diss-adu-simple-op|train",
            "m112{}-diss-adu-simple-op|test",
        ],
        ("adu", FULL_RELATION_SET_ADU, "en", "m112en"): [
            "m112{}-diss-adu-full-baseline-first",
            "m112{}-diss-adu-full-baseline-prec",
            "m112{}-diss-adu-full-op|equal",
            "m112{}-diss-adu-full-op|train",
            "m112{}-diss-adu-full-op|test",
        ],
        # en edu
        ("edu", SIMPLE_RELATION_SET, "en", "m112en_fine"): [
            "m112{}-diss-edu-simple-baseline-first",
            "m112{}-diss-edu-simple-baseline-prec",
            "m112{}-diss-edu-simple-op|equal",
            "m112{}-diss-edu-simple-op|train",
            "m112{}-diss-edu-simple-op|test",
        ],
        ("edu", FULL_RELATION_SET, "en", "m112en_fine"): [
            "m112{}-diss-edu-full-baseline-first",
            "m112{}-diss-edu-full-baseline-prec",
            "m112{}-diss-edu-full-op|equal",
            "m112{}-diss-edu-full-op|train",
            "m112{}-diss-edu-full-op|test",
        ],
    }

    for (
        (segmentation, relationset, lang, corpus_id),
        conditions,
    ) in settings.iteritems():
        conditions = [c.format(lang) for c in conditions]
        evaluate_setting(
            lang, segmentation, relationset, conditions, corpus_id=corpus_id
        )
