from typing import Any
from observational_fear import load
from observational_fear.events import get_block_starts
from observational_fear.align import align_freeze, align_block
from observational_fear.config import BlockConfig, FreezeConfig, Config
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd
from neurobox.wide_transforms import resample



class TraceHandler:
    def __init__(self, config: Any):
        self.config = config
        self.is_freeze = config.is_freeze
        self._traces_wide= None
        self._traces_long = None
        self._events = None
        self._aligned = None
        self._aligned_wide = None
        self._df_cells =  load.load_cells(self.config.data_dir).rename(columns={"mouse": "mouse_name"}).assign(new_id=lambda x: x["new_id"].astype(str))
        self._coreg_cells = load.load_block_coreg_cells(self.config.data_dir)


    def rotate_traces(self, df):
        df = df.set_index("time")
        rotated_vals = np.roll(df, np.random.randint(0, len(df) - 1), axis=0)
        return pd.DataFrame(rotated_vals, columns=df.columns, index=df.index).reset_index()

    @property
    def traces_wide(self):
        if self._traces_wide is None:
            traces_wide = load.load_traces(data_dir=self.config.data_dir, session=self.config.session)
            self._traces_wide = resample(traces_wide.set_index("time"), new_interval=self.config.sampling_interval).reset_index()
            if self.config.coreg_only:
                self._traces_wide = self._traces_wide[[c for c in self._traces_wide if c in self._coreg_cells or c == "time"]]
            if self.config.normalize:
                normalized = self._traces_wide.set_index("time")
                vals = MinMaxScaler().fit_transform(normalized)
                self._traces_wide = pd.DataFrame(vals, columns=normalized.columns, index=normalized.index).reset_index()
        return self._traces_wide

    @property
    def traces_long(self):
        if self._traces_long is None:
            self._traces_long = self._wide_to_long(self.traces_wide)
        return self._traces_long


    @property
    def events(self):
        if self._events is None:
            self._events = self._get_events(self.config)
        return self._events
    
    @property
    def aligned(self):
        if self._aligned is None:
            self._aligned = self._align(self.traces_long, self.events, self.config)
        return self._aligned
    
    @property
    def aligned_wide(self):
        if self._aligned_wide is None:
            self._aligned_wide = self.trial_pivot(self.aligned)
        return self._aligned_wide

    
    def _wide_to_long(self, df):
        return df.melt(id_vars=["time"]).merge(self._df_cells)
    
    def _get_events(self, config):
        if self.is_freeze:
            events = load.get_freeze_events(self.config.data_dir, config.session, config.start_stop, 
            role=config.role, min_freeze_diff=config.min_freeze_diff)
        else:
            events = get_block_starts(config.session, block_name=config.block)
        return events


    def _align(self, traces, events, config):
        preprocess_func = self._align_freeze if self.is_freeze else self._align_block
        return preprocess_func(traces.copy(), events, config)
    
    def _align_freeze(self, traces, events, config):
        aligned = align_freeze(traces, events, config.t_before, config.t_after)
        aligned = aligned.drop(["group", "relative_to_event"], axis=1)
        return aligned

    
    def _align_block(self, traces, events, config):
        aligned = align_block(traces, events, t_before=config.t_before, t_after=config.t_after)
        return aligned

    
    def generate_rotated(self, use_cache: bool=True):
        traces_wide = self.rotate_traces(self.traces_wide.copy())
        traces_long = self._wide_to_long(traces_wide)
        if use_cache:
            aligned = traces_long.reindex(self.aligned.index).assign(aligned=self.aligned["aligned"], trial=self.aligned["trial"])
        else:
            aligned = self._align(traces_long, self.events, self.config)
        return aligned
    
    @staticmethod
    def trial_pivot(df, fillna=True):
        df_piv = df.pivot(index=["trial", "aligned",], columns="new_id", values="value")
        if fillna:
            df_piv = df_piv.fillna(method="ffill").fillna(method="bfill")
        return df_piv

