from typing import List, Tuple
from revdict.paths import REVDICT_DATASET_TSV
import csv


def load_revdict_dataset() -> List[List[str]]:
    """
    (en, kr, lang, def)
    :return:
    """
    with open(REVDICT_DATASET_TSV, 'r') as fh:
        tsv_reader = csv.reader(fh, delimiter="\t")
        # skip the header
        next(tsv_reader)
        return [
            row
            for row in tsv_reader
        ]
