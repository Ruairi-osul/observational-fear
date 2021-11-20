"""
Transforms for plugging into neurobox pipelines 

Meaning all functions are of the form Callable[[pd.DataFrame, ...], pd.DataFrame]
"""

import pandas as pd
from .preprocessing import get_blocknames_trialnumber


def get_exp_phase(
    df: pd.DataFrame, session: str, time_col: str = "time"
) -> pd.DataFrame:
    block_names, trial_number = get_blocknames_trialnumber(
        df[time_col].values, session=session
    )
    return df.assign(block=block_names, trial_number=trial_number)
