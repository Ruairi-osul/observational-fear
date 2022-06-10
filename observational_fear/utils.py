from observational_fear import load
from observational_fear.nbox_transforms import align_to_data_by
import pandas as pd
from typing import Optional
from observational_fear.events import get_block_starts
from binit.align import align_around
from binit.bin import which_bin_idx
from pathlib import Path

def align_to_freeze(
    data_dir,
    session,
    t_before,
    t_after,
    start_stop,
    role="obs",
    rotate=False,
    min_freeze_diff=11,
    sampling_interval="100ms",
    coreg_only=True,
):
    df_cells = load.load_cells(data_dir).rename(columns={"mouse": "mouse_name"})
    df_cells["new_id"] = df_cells["new_id"].astype("str")
    if start_stop == "start":
        freeze_getter = load.load_freeze_starts
        freeze_col = "freeze_start"
    elif start_stop == "stop":
        freeze_getter = load.load_freeze_stops
        freeze_col = "freeze_stop"
    events = freeze_getter(data_dir, session, role=role, min_diff=min_freeze_diff)
    traces = load.load_traces_long(data_dir, session, rotate=rotate, sampling_interval=sampling_interval, coreg_only=coreg_only).merge(df_cells)
    df_aligned = align_to_data_by(
        df_data=traces,
        df_events=events,
        df_data_time_col="time",
        df_data_cell_col="new_id",
        df_data_value_col="value",
        time_before_event=t_before,
        time_after_event=t_after,
        df_data_group_col="mouse_name",
        df_events_timestamp_col=freeze_col,
        df_events_group_colname="mouse_name",
        precision=1,
    )
    return df_aligned


def align_to_block(
    df_traces: pd.DataFrame,
    session: str,
    block: str,
    t_before: float = 5,
    t_after: float = 5,
    first_trial: Optional[int] = None,
    last_trial: Optional[int] = None,
):
    """Average trace around block

    Args:
        df_traces (pd.DataFrame): Traces in long format
        session (str): Session Name
        block (str): Block Name
        t_before (float, optional): Align time before event. Defaults to 5.
        t_after (float, optional): Align time after event. Defaults to 5.
        standardize (bool, optional): Zscore or not. On aligned data. Defaults to True.
        first_trial (Optional[int], optional): What trial to start. Defaults to first.
        last_trial (Optional[int], optional): What trial to stop. Defaults to last.
    """
    events = get_block_starts(session=session, block_name=block)
    df_traces["aligned"] = align_around(
        df_traces["time"].values, events, t_before=t_before, max_latency=t_after
    ).round(1)
    df_traces = df_traces.dropna()
    df_traces["trial"] = which_bin_idx(
        df_traces["time"].values, events, time_before=t_before, time_after=t_after
    )
    if first_trial is not None:
        df_traces = df_traces[df_traces["trial"] >= first_trial]
    if last_trial is not None:
        df_traces = df_traces[df_traces["trial"] <= last_trial]
    return df_traces


def load_traces_aligned_to_block_wide(
    data_dir: Path,
    session: str,
    block: str,
    t_before=10,
    t_after=10,
    sampling_interval="100ms",
    coreg_only=True,
    rotate=False
):
    traces = load.load_traces_long(
        data_dir=data_dir,
        session=session,
        sampling_interval=sampling_interval,
        coreg_only=coreg_only,
        rotate=rotate
    )
    aligned = align_to_block(
        traces, session=session, block=block, t_before=t_before, t_after=t_after
    ).copy()
    aligned["session"] = session
    return aligned.pivot(
        index=["session", "trial", "aligned"], columns="new_id", values="value"
    )

def load_traces_aligned_to_freeze_wide(
    data_dir: Path,
    session: str,
    start_stop: str,
    t_before=10,
    t_after=10,
    sampling_interval="100ms",
    coreg_only=True,
    rotate=False,
    min_freeze_diff=11,
):
    traces = align_to_freeze(
        data_dir, 
        session, 
        t_before=t_before, 
        t_after=t_after, 
        start_stop=start_stop,
        rotate=rotate,
        min_freeze_diff=min_freeze_diff, 
        sampling_interval=sampling_interval, 
        coreg_only=coreg_only)

    return (
        traces
        .rename(columns={"event": "trial"})
        .loc[lambda x: x.trial < x.groupby("mouse_name").trial.max().min()]
        .copy()
        .assign(start_stop=start_stop)
        .pivot(["start_stop", "trial", "aligned"], columns="new_id", values="value")
        .fillna(method="ffill")
        .fillna(method="bfill")
    )


class WideLoader:
    def __init__(self, freeze: bool=False, **kwargs):
        self._traces = None
        self.freeze_loader = load_traces_aligned_to_freeze_wide
        self.block_loader = load_traces_aligned_to_block_wide
        self.loader = self.freeze_loader if freeze else self.block_loader
        self.kwargs = kwargs

    @property
    def traces(self):
        if self._traces is None:
            self._traces = self.loader(**self.kwargs)
        return self._traces