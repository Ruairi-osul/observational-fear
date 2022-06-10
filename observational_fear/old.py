import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
from observational_fear.utils import (
    load_traces_aligned_to_block_wide,
    load_traces_aligned_to_freeze_wide,
)
from scipy.stats import zscore
from pathlib import Path
from scipy.spatial import distance
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Callable, Optional, Tuple


def get_scree(X):
    mod = PCA()
    mod.fit(X)
    explained_variance = (1 - mod.explained_variance_ratio_) * 100
    num_pcs = np.arange(len(explained_variance)) + 1
    return pd.DataFrame({"Num_PCs": num_pcs, "Var_retained": explained_variance})


def get_loadings(neurons, mod):
    loadings = mod.components_.T * np.sqrt(mod.explained_variance_)
    return pd.DataFrame(
        {"neuron_id": neurons, "PC1": loadings[:, 0], "PC2": loadings[:, 1]}
    )


class TrajectorySolver:
    def __init__(self, model: Optional[Callable] = None):

        self._trajectories = None
        self._mod = None
        self._average_trajectories = None
        self._average_traces = None

        self.model = PCA(2) if model is None else model

    @property
    def scree(self):
        return get_scree(self.traces)

    @property
    def loadings(self):
        return get_loadings(self.traces.columns, self.mod)

    @property
    def average_traces(self):
        if self._average_traces is None:
            self._average_traces = (
                self.traces.reset_index().groupby([self.compare_col, "aligned"]).mean()
            )
        return self._average_traces

    @staticmethod
    def _get_difference_from_centroid(
        df_trial, centroid, returned_colname="distance_from_centroid"
    ):
        df_trial[returned_colname] = df_trial[["PC1", "PC2"]].apply(
            lambda x: distance.euclidean(x.values, centroid), axis=1
        )
        return df_trial

    def _calculate_centroid_distance(self, trajectories):
        centroid_pre = (
            trajectories.loc[lambda x: x.aligned < 0][["PC1", "PC2"]].mean().values
        )
        centroid_post = (
            trajectories.loc[lambda x: x.aligned >= 0][["PC1", "PC2"]].mean().values
        )
        trajectories = trajectories.groupby([self.compare_col]).apply(
            self._get_difference_from_centroid,
            centroid=centroid_pre,
            returned_colname="distance_from_pre",
        )
        trajectories = trajectories.groupby([self.compare_col]).apply(
            self._get_difference_from_centroid,
            centroid=centroid_post,
            returned_colname="distance_from_post",
        )
        return trajectories


class TrajectoryAvg(TrajectorySolver):
    @property
    def trajectories(self):
        if self._trajectories is None:
            self.model.fit(self.average_traces)
            transformed = self.model.transform(self.traces)
            self._trajectories = (
                pd.DataFrame(
                    transformed, index=self.traces.index, columns=["PC1", "PC2"]
                )
                .reset_index()
                .assign(prepost=lambda x: np.where(x["aligned"] < 0, "pre", "post"))
            )
        return self._trajectories

    @property
    def average_trajectories(self):
        if self._average_trajectories is None:
            transformed = self.model.transform(self.average_traces)
            self._average_trajectories = (
                pd.DataFrame(
                    transformed, index=self.traces.index, columns=["PC1", "PC2"]
                )
                .reset_index()
                .assign(prepost=lambda x: np.where(x["aligned"] < 0, "pre", "post"))
            )
            self._average_trajectories = self._calculate_centroid_distance(
                self._average_trajectories
            )
        return self._average_trajectories


class TrajectoryByTrial(TrajectorySolver):
    @property
    def trajectories(self):
        if self._trajectories is None:
            self.model.fit(self.traces)
            transformed = self.model.transform(self.traces)
            self._trajectories = (
                pd.DataFrame(
                    transformed, index=self.traces.index, columns=["PC1", "PC2"]
                )
                .reset_index()
                .assign(prepost=lambda x: np.where(x["aligned"] < 0, "pre", "post"))
            )
        return self._trajectories

    @property
    def average_trajectories(self):
        if self._average_trajectories is None:
            trajectories = (
                self.trajectories.groupby([self.compare_col, "aligned"])[["PC1", "PC2"]]
                .mean()
                .reset_index()
                .assign(prepost=lambda x: np.where(x["aligned"] < 0, "pre", "post"))
            )
            self._average_trajectories = self._calculate_centroid_distance(trajectories)
        return self._average_trajectories


class TrajectoryFinder:
    def __init__(
        self,
        solver: str,
        data_dir: Path,
        t_before: int,
        t_after: int,
        rotate: bool = False,
        sampling_interval: str = "100ms",
        coreg_only: bool = False,
        save_dir: Optional[Path] = None,
        **kwargs,
    ):
        if solver == "Average":
            self.solver: TrajectorySolver = TrajectoryAvg(**kwargs)
        elif solver == "ByTrial":
            self.solver = TrajectoryByTrial(**kwargs)
        else:
            raise ValueError("Unknown Solver: Enter one of {'Average', 'ByTrial'}")

        self.data_dir = data_dir
        self.sampling_interval = sampling_interval
        self.t_before = t_before
        self.t_after = t_after
        self.rotate = rotate
        self._traces = None
        self.coreg_only = coreg_only
        self.save_dir = save_dir if save_dir is not None else Path()
        self.solver.traces = self.traces

    @property
    def scree(self):
        return self.solver.scree

    @property
    def trajectories(self):
        return self.solver.trajectories

    @property
    def average_trace(self):
        return self.seolver.average_trace

    @property
    def average_trajectory(self):
        return self.solver.average_trajectory

    @property
    def loadings(self):
        return self.solver.loadings

    def save(self):
        self.trajectories.to_csv(
            self.save_dir / f"{self.base_name} - Trajectory Time Series.csv",
            index=False,
        )
        self.loadings.to_csv(
            self.save_dir / f"{self.base_name} - Loadings.csv", index=False
        )
        self.scree.to_csv(self.save_dir / f"{self.base_name} - Scree.csv", index=False)


class TrajectoryBlock(TrajectoryFinder):
    compare_col = "session"

    def __init__(
        self, session: str, block: str, **kwargs,
    ):
        super().__init__(**kwargs)
        self.session = session
        self.block = block
        self.base_name = f"Trajectories - Freeze - {self.session} - {self.block} - "

    @property
    def traces(self):
        if self._traces is None:
            self._traces = load_traces_aligned_to_block_wide(
                self.data_dir,
                session=self.session,
                block=self.block,
                sampling_interval=self.sampling_interval,
                t_before=self.t_before,
                t_after=self.t_after,
                rotate=self.rotate,
                coreg_only=self.coreg_only,
            ).apply(zscore)
        return self._traces


class TrajectoryFreeze(TrajectoryFinder):
    compare_col = "start_stop"

    def __init__(self, session: str, start_stop: str, **kwargs):
        super().__init__(**kwargs)
        self.session = session
        self.start_stop = start_stop
        self.base_name = (
            f"Trajectories - Freeze - {self.session} - {self.start_stop} - "
        )

    @property
    def traces(self):
        if self._traces is None:
            self._traces = load_traces_aligned_to_freeze_wide(
                self.data_dir,
                session=self.session,
                start_stop=self.start_stop,
                sampling_interval=self.sampling_interval,
                t_before=self.t_before,
                t_after=self.t_after,
                rotate=self.rotate,
                coreg_only=self.coreg_only,
            ).apply(zscore)
        return self._traces


class TrajectoryBlockCombined(TrajectoryFinder):
    compare_col = "session"

    def __init__(
        self, session_block1: Tuple[str, str], session_block2: Tuple[str, str], **kwargs
    ):
        super().__init__(**kwargs)
        self.session_block1 = session_block1
        self.session_block2 = session_block2
        self.base_name = f"Trajectories - Combined {session_block1[0]} {session_block1[1]} - {session_block1[1]} {session_block1[1]} - "

    @property
    def traces(self):
        if self._traces is None:
            traces_1 = load_traces_aligned_to_block_wide(
                self.data_dir,
                session=self.session_block1[0],
                block=self.session_block1[1],
                sampling_interval=self.sampling_interval,
                t_before=self.t_before,
                t_after=self.t_after,
                rotate=self.rotate,
                coreg_only=self.coreg_only,
            )
            traces_2 = load_traces_aligned_to_block_wide(
                self.data_dir,
                session=self.session_block2[0],
                block=self.session_block2[1],
                sampling_interval=self.sampling_interval,
                t_before=self.t_before,
                t_after=self.t_after,
                rotate=self.rotate,
                coreg_only=self.coreg_only,
            )
            self._traces = pd.concat([traces_1, traces_2]).apply(zscore)
        return self._traces


class TrajectoryFreezeCombined(TrajectoryFinder):
    compare_col = "start_stop"

    def __init__(self, session: str, **kwargs):
        super().__init__(**kwargs)
        self.session = session
        self.base_name = (
            f"Trajectories - Freeze - Combined 0 {self.session} - StartStop - "
        )

    @property
    def traces(self):
        if self._traces is None:
            start = load_traces_aligned_to_freeze_wide(
                self.data_dir,
                session=self.session,
                start_stop="start",
                sampling_interval=self.sampling_interval,
                t_before=self.t_before,
                t_after=self.t_after,
                rotate=self.rotate,
                coreg_only=self.coreg_only,
            )
            stop = load_traces_aligned_to_freeze_wide(
                self.data_dir,
                session=self.session,
                start_stop="stop",
                sampling_interval=self.sampling_interval,
                t_before=self.t_before,
                t_after=self.t_after,
                rotate=self.rotate,
                coreg_only=self.coreg_only,
            )
            self._traces = pd.concat([start, stop]).apply(zscore)
        return self._traces


class TrajectoryVisulizer:
    def __init__(self, name, save_dir=None, save=False):
        self.name = name
        self.save_dir = save_dir if save_dir is not None else Path()
        self.save = save

    def plot_loadings(
        self, loadings,
    ):
        plt.figure()
        plt.scatter(loadings["PC1"], loadings["PC2"], color="black")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(self.name)
        if self.save:
            plt.savefig(self.save_dir / f"Trajectories - {self.name} - Loadings.svg")
        return plt.gca()

    def plot_scree(self, scree):
        plt.figure()
        plt.plot(scree["Num_PCs"], scree["Var_retained"])
        plt.xlabel("Num_PCs")
        plt.ylabel("Variance Retained")
        plt.title(self.name)
        return plt.gca()

    def plot_trajectories_pc_space_2d(
        self, average_trajectories, cmap="seismic", compare_col="session"
    ):
        plt.figure()
        for session in average_trajectories[compare_col].unique():
            df_session = average_trajectories.loc[lambda x: x[compare_col] == session]
            for prepost, marker in zip(["pre", "post"], ["o", "+"]):
                df_prepost = df_session.loc[lambda x: x.prepost == prepost]
                plt.scatter(
                    df_prepost["PC1"],
                    df_prepost["PC2"],
                    c=df_prepost["aligned"],
                    cmap=cmap,
                    label=f"{session} {prepost}",
                    marker=marker,
                    vmin=average_trajectories["aligned"].min(),
                    vmax=average_trajectories["aligned"].max(),
                )
        plt.colorbar()
        plt.legend()
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        if self.save:
            plt.savefig(self.save_dir / f"Trajectories - {self.name} - 2DScatter.svg")
        return plt.gca()

    def plot_trajectories_pc_space_3d(
        self, trajectories, avg_only=True, rotation=None, compare_col="session"
    ):
        _, ax = plt.subplots(subplot_kw={"projection": "3d"})
        if rotation is not None:
            ax.view_init(*rotation)

        colors = ["black", "blue"]
        for i, session in enumerate(trajectories[compare_col].unique()):
            df_session = trajectories.loc[lambda x: x[compare_col] == session]
            if not avg_only:
                for trial in df_session["trial"].unique():
                    dfp = df_session[df_session["trial"] == trial].sort_values(
                        "aligned"
                    )
                    ax.plot(
                        dfp["aligned"],
                        dfp["PC1"],
                        dfp["PC2"],
                        alpha=0.2,
                        color=colors[i],
                    )

            dfp = df_session.groupby(["aligned"], as_index=False)[["PC1", "PC2"]].mean()
            dfp = dfp.sort_values("aligned")
            ax.plot(
                dfp["aligned"],
                dfp["PC1"],
                dfp["PC2"],
                linewidth=2,
                color=colors[i],
                label=session,
            )

        ax.set_xlabel("Time")
        ax.set_ylabel("PC1")
        ax.set_zlabel("PC2")
        if self.save:
            plt.savefig(self.save_dir / f"Trajectories - {self.name} - 3D.svg")
        return ax

    def plot_trajectories_line(self, trajectories, hue=None):
        _, axes = plt.subplots(nrows=2, sharex=True)
        sns.lineplot(
            data=trajectories, x="aligned", y="PC1", ci="sd", hue=hue, ax=axes[0]
        )
        sns.lineplot(
            data=trajectories, x="aligned", y="PC2", ci="sd", hue=hue, ax=axes[1]
        )
        plt.xlabel("Time Relative to US")
        if self.save:
            plt.savefig(
                self.save_dir / f"Trajectories - {self.name} - TRIAL PC Line plots.svg"
            )
        return axes

    def plot_avg_trajectories_line(self, trajectories, compare_col="session"):
        _, axes = plt.subplots(nrows=2, sharex=True)
        for session in trajectories[compare_col].unique():
            t_sub = trajectories.loc[lambda x: x[compare_col] == session]
            axes[0].scatter(x=t_sub["aligned"], y=t_sub["PC1"], label=session)
            axes[0].set_ylabel("PC1")
            axes[1].scatter(x=t_sub["aligned"], y=t_sub["PC2"], label=session)
            axes[1].set_ylabel("PC2")
        plt.xlabel("Time Relative to US")
        plt.legend()
        if self.save:
            plt.savefig(
                self.save_dir / f"Trajectories - {self.name} - AVG PC Line plots.svg"
            )
        return axes
