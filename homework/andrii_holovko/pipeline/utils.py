from functools import reduce
from typing import Dict, Any, Sequence

import pandas as pd


def least_populated_class(data: Sequence[pd.Series]) -> int:
    return reduce(lambda x, y: x + '\n\n' + y, data).value_counts().min()


def to_text(d: Dict[str, Any]) -> str:
    s = '\n\n'
    for k, v in d.items():
        s += k + f':\n\t{v}\n\n'
    return s
