from neurobox.long_transforms import align_to_events, get_closest_event_idx
import pandas as pd
import numpy as np
from typing import Optional
from binit.bin import which_bin_idx
from binit.align import align_around

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
        time_before=time_before_event,
        time_after=time_after_event,
        df_data_group_colname=df_data_group_col,
        df_events_group_colname=df_events_group_colname,
        df_events_timestamp_col=df_events_timestamp_col,
        df_data_time_col=df_data_time_col,
        returned_colname="event",
    )
    df["relative_to_event"] = np.where(df["aligned"] < 0, "pre_event", "post_event")
    return df



def align_freeze(traces, events, t_before, t_after, trial_col="trial"):
    preprocessed = align_to_data_by(
        df_data=traces,
        df_events=events,
        df_data_time_col="time",
        df_data_cell_col="new_id",
        df_data_value_col="value",
        time_before_event=t_before,
        time_after_event=t_after,
        df_data_group_col="mouse_name",
        df_events_timestamp_col=trial_col,
        df_events_group_colname="mouse_name",
        precision=1,
    ).rename(columns={"event": trial_col})
    max_event = preprocessed.groupby("mouse_name")[trial_col].max().min()
    return preprocessed.loc[lambda x: x[trial_col] < max_event]



def align_block(traces, events, t_before, t_after):
    traces["aligned"] = align_around(
        traces["time"].values, events, t_before=t_before, max_latency=t_after
    ).round(1)
    traces = traces.dropna()
    traces["trial"] = which_bin_idx(
        traces["time"].values, events, time_before=t_before, time_after=t_after
    )
    return traces