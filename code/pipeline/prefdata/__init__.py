"""Preference datasets: loaders that normalize real pairwise-preference data
to the pipeline's record shape.

The normalized record shape every loader emits (and `stages/data_prep.py`
writes to `prefs.jsonl`) is:

    {"prompt": str, "chosen": str, "rejected": str, "source": str, ...meta}

Parsing is separated from fetching: `normalize_*` functions are pure
row-to-record transforms unit-tested offline on fixture dicts; `load_*`
functions do the network fetch via `datasets` (imported lazily, because this
package must import cleanly in the offline test environment).
"""

from pipeline.prefdata.loaders import (  # noqa: F401
    LOADERS,
    TEST_SPLITS,
    TRAIN_SPLITS,
    load_records,
    normalize_hh,
    normalize_ultrafeedback,
    subsample,
    within_chars,
)
