from evidencegraph.argtree import SIMPLE_RELATION_SET
from evidencegraph.evaluation import evaluate_setting


if __name__ == "__main__":
    lang = "en"

    settings = {
        ("adu", SIMPLE_RELATION_SET): [
            "argmin2018-normal-m112en-adu-simple-op|equal",
            "argmin2018-cross-part2+m112en-adu-simple-op|equal",
            "argmin2018-add-part2+m112en-adu-simple-op|equal",
        ]
    }

    for (segmentation, relationset), conditions in settings.iteritems():
        evaluate_setting(
            lang, segmentation, relationset, conditions, corpus_id="m112en"
        )

    settings = {
        ("adu", SIMPLE_RELATION_SET): [
            "argmin2018-normal-part2-adu-simple-op|equal",
            "argmin2018-cross-m112en+part2-adu-simple-op|equal",
            "argmin2018-add-m112en+part2-adu-simple-op|equal",
        ]
    }

    for (segmentation, relationset), conditions in settings.iteritems():
        evaluate_setting(
            lang,
            segmentation,
            relationset,
            conditions,
            corpus_id="m112en_part2",
        )
