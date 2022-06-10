from pathlib import Path
import pandas as pd
import numpy as np
from neurobox.wide_transforms import resample
from typing import Optional

from pyrsistent import freeze
from observational_fear.events import get_freeze_starts, get_freeze_stops


def load_cells(data_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(data_dir / "cells.parquet.gzip")


def load_cell_mapper(data_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(data_dir / "cells_mapper.parquet.gzip")


def load_behaviour(data_dir: Path, session: str) -> pd.DataFrame:
    return pd.read_parquet(data_dir / f"{session}-behaviour.parquet.gzip")


def load_traces(data_dir: Path, session: str) -> pd.DataFrame:
    return pd.read_parquet(data_dir / f"{session}-traces.parquet.gzip").fillna(method="ffill").fillna(0)


def load_freeze(data_dir: Path, session: str) -> pd.DataFrame:
    return pd.read_parquet(data_dir / f"{session}-freeze.parquet.gzip")


def load_freeze_starts(
    data_dir: Path, session: str, role: str = "obs", min_diff: Optional[float] = 11
):
    behaviour = load_behaviour(data_dir=data_dir, session=session)
    if session == "day2":
        behaviour = behaviour.loc[lambda x: x.role == role]
    freeze_starts = get_freeze_starts(df=behaviour, freeze_col="was_freezing")
    if min_diff is not None:
        freeze_starts = (
            freeze_starts.groupby("mouse_name")["freeze_start"]
            .apply(lambda x: x[x.diff() > 10])
            .reset_index()
            .drop("level_1", axis=1)
        )
    return freeze_starts


def load_freeze_stops(
    data_dir: Path, session: str, role: str = "obs", min_diff: Optional[float] = 11
) -> pd.DataFrame:
    behaviour = load_behaviour(data_dir=data_dir, session=session)
    if session == "day2":
        behaviour = behaviour.loc[lambda x: x.role == role]
    freeze_stops = get_freeze_stops(df=behaviour, freeze_col="was_freezing")
    if min_diff is not None:
        freeze_stops = (
            freeze_stops.groupby("mouse_name")["freeze_stop"]
            .apply(lambda x: x[x.diff() > 10])
            .reset_index()
            .drop("level_1", axis=1)
        )    
    return freeze_stops


def load_block_coreg_cells(data_dir: Path) -> np.ndarray:
    return (
        pd.read_csv(data_dir / "derived" / "coregistered_nofreeze.csv")
        .new_id.unique()
        .astype(str)
    )



def load_traces_long(
    data_dir: Path,
    session: str,
    coreg_only: bool = True,
    sampling_interval: str = "100ms",
    rotate: bool = False,
) -> pd.DataFrame:

    df = load_traces(data_dir, session=session)

    df = resample(df.set_index("time"), sampling_interval).reset_index()

    if rotate:
        df = df.set_index("time_col")
        df = (
            pd.DataFrame(
            data=np.roll(df, np.random.randint(0, len(df) - 1), axis=0), 
            columns=df.columns, index=df.index).reset_index()
            )

    df = df.melt(id_vars=["time"])
    if coreg_only:
        coreg_cells = load_block_coreg_cells(data_dir)
        df = df[df["new_id"].isin(coreg_cells)]
    return df


def get_freeze_events(data_dir, session, start_stop, min_freeze_diff=11, role="obs"):
    if start_stop == "start":
        freeze_getter = load_freeze_starts
        freeze_col = "freeze_start"
    elif start_stop == "stop":
        freeze_getter = load_freeze_stops
        freeze_col = "freeze_stop"
    events = freeze_getter(
        data_dir, session, min_diff=min_freeze_diff, role=role, 
        ).rename(columns={freeze_col: "trial"})
    return events
