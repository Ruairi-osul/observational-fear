"""
Transforms for plugging into neurobox pipelines 

Meaning all functions are of the form Callable[[pd.DataFrame, ...], pd.DataFrame]
"""

import pandas as pd
from .preprocessing import get_blocknames_trialnumber
import numpy as np
from typing import Callable, Optional
from binit.bin import which_bin_idx
from binit.align import align_around
from .stats import p_adjust
from scipy.stats import wilcoxon
from neurobox.long_transforms import align_to_events, get_closest_event_idx


def get_exp_phase(
    df: pd.DataFrame, session: str, time_col: str = "time"
) -> pd.DataFrame:
    block_names, trial_number = get_blocknames_trialnumber(
        df[time_col].values, session=session
    )
    return df.assign(block=block_names, trial_number=trial_number)


def summarize_prepost_events(
    df: pd.DataFrame,
    events: np.ndarray,
    t_before: float,
    t_after: float,
    summary_func: Callable[[np.ndarray], np.ndarray],
    precision: Optional[float] = None,
    cell_col: str = "neuron_id",
    time_col: str = "time",
    value_col: str = "value",
) -> pd.DataFrame:
    df = df[[cell_col, time_col, value_col]].copy()
    df["_aligned"] = align_around(
        df[time_col].values, events, t_before=t_before, max_latency=t_after
    )
    if precision is not None:
        df["_aligned"] = df["_aligned"].round(precision)
    df = df.dropna()
    df["_prepost"] = np.where(df["_aligned"] < 0, "pre_event", "post_event")
    df["_trial"] = which_bin_idx(
        df[time_col], bin_edges=events, time_before=t_before, time_after=t_after,
    )
    return (
        df.groupby(["_trial", "_prepost", cell_col], as_index=False)[value_col]
        .apply(summary_func)
        .pivot(index=["_trial", cell_col], columns=["_prepost"], values=value_col)
        .reset_index()
        .rename_axis(None, axis=1)
    )


def wilcoxon_by_cell(
    df: pd.DataFrame,
    cell_col: str,
    pre_event_col: str = "pre_event",
    post_event_col: str = "post_event",
) -> pd.DataFrame:
    def _wilcoxon(df: pd.DataFrame) -> pd.Series:
        stat, p = wilcoxon(df[pre_event_col], df[post_event_col])
        diff_medians = df[pre_event_col].median() - df[post_event_col].median()
        return pd.Series({"statistic": stat, "p": p, "diff_of_medians": diff_medians})

    df = df.groupby(cell_col).apply(_wilcoxon)
    df["p"] = p_adjust(df["p"])
    return df


def align_to_data_by(
    df_data: pd.DataFrame,
    df_events: pd.DataFrame,
    df_data_cell_col: str,
    df_data_group_col: str,
    df_events_group_colname: str,
    df_events_timestamp_col: str,
    time_before_event: int,
    time_after_event: int,
    df_data_time_col: str = "time",
    df_data_value_col: str = "value",
    precision: Optional[int] = None,
) -> pd.DataFrame:
    df = df_data[
        [df_data_time_col, df_data_cell_col, df_data_value_col, df_data_group_col]
    ].copy()
    df = align_to_events(
        df_data=df,
        df_events=df_events,
        time_before_event=time_before_event,
        max_latency=time_after_event,
        df_data_group_colname=df_data_group_col,
        df_events_group_colname=df_events_group_colname,
        df_events_timestamp_col=df_events_timestamp_col,
        df_data_time_col=df_data_time_col,
        returned_colname="aligned",
    )
    df.dropna(inplace=True)
    if precision:
        df["aligned"] = df["aligned"].round(precision)
    df = get_closest_event_idx(
        df_data=df,
        df_events=df_events,
        time_before_event=time_before_event,
        max_latency=time_after_event,
        df_data_group_colname=df_data_group_col,
        df_events_group_colname=df_events_group_colname,
        df_events_timestamp_col=df_events_timestamp_col,
        df_data_time_col=df_data_time_col,
        returned_colname="event",
    )
    df["relative_to_event"] = np.where(df["aligned"] < 0, "pre_event", "post_event")
    return df


def exclude_short_trials(
    df: pd.DataFrame,
    min_bins_pre_event: int,
    min_bins_post_event: int,
    cell_col: str,
    trial_col: str,
    aligned_time_col: str = "aligned",
) -> pd.DataFrame:
    df["_prepost"] = np.where(df[aligned_time_col] < 0, "pre_event", "post_event")
    return df.groupby([trial_col, cell_col]).filter(
        lambda x: len(x.loc[x["_prepost"] == "pre_event"]) > min_bins_pre_event
        and len(x.loc[x["_prepost"] == "post_event"]) > min_bins_post_event
    )


def summarize_prepost(
    df: pd.DataFrame,
    trial_col: str,
    summary_func: Callable[[np.ndarray], np.ndarray],
    cell_col: str,
    value_col: str = "value",
    event_col: str = "event",
    aligned_time_col: str = "aligned",
) -> pd.DataFrame:
    df["_prepost"] = np.where(df[aligned_time_col] < 0, "pre_event", "post_event")
    return (
        df.groupby([trial_col, "_prepost", cell_col], as_index=False)[value_col]
        .apply(summary_func)
        .pivot(index=[event_col, cell_col], columns=["_prepost"], values=value_col)
        .reset_index()
        .rename_axis(None, axis=1)
    )
