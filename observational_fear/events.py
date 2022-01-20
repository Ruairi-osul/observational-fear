import numpy as np
import pandas as pd
from .preprocessing import get_blocks


def get_block_starts(session: str, block_name: str) -> np.ndarray:
    _, blocks = get_blocks(session)
    block_starts = np.cumsum([0] + [block.length for block in blocks])
    return np.array(
        [
            block_start
            for i, block_start in enumerate(block_starts[:-1])
            if blocks[i].name == block_name
        ]
    )


def get_freeze_starts(
    df: pd.DataFrame,
    freeze_col: str,
    mouse_col: str = "mouse_name",
    time_col: str = "time",
):
    return (
        df.groupby(mouse_col)
        .apply(lambda x: x.loc[x[freeze_col].diff() == 1][time_col].values)
        .explode()
        .reset_index()
        .rename(columns={0: "freeze_start"})
        .assign(freeze_start=lambda x: x["freeze_start"].astype(float))
    )


def get_freeze_stops(
    df: pd.DataFrame,
    freeze_col: str,
    mouse_col: str = "mouse_name",
    time_col: str = "time",
):
    return (
        df.groupby(mouse_col)
        .apply(lambda x: x.loc[x[freeze_col].diff() == -1][time_col].values)
        .explode()
        .reset_index()
        .rename(columns={0: "freeze_stop"})
        .assign(freeze_stop=lambda x: x["freeze_stop"].astype(float))
    )
