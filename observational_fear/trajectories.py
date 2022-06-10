import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler

from pathlib import Path
from scipy.spatial import distance
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Callable, Optional


def get_scree(mod):
    explained_variance = (1 - mod.explained_variance_ratio_) * 100
    num_pcs = np.arange(len(explained_variance)) + 1
    return pd.DataFrame({"Num_PCs": num_pcs, "Var_retained": explained_variance})


def get_loadings(neurons, mod):
    loadings = mod.components_.T * np.sqrt(mod.explained_variance_)
    return pd.DataFrame(
        {"neuron_id": neurons, "PC1": loadings[:, 0], "PC2": loadings[:, 1]}
    )


class TrajectorySolver:
    def __init__(self, method, compare_col: str, traces: pd.DataFrame, centerer: Optional[Callable]=None, model: Optional[Callable] = None):
        self.method = method
        self._trajectories_getter = self.which_trajectories(self.method)
        self._average_trajectories_getter = self.which_average_trajectories(self.method)
        self._trajectories = None
        self._mod = None
        self._average_trajectories = None
        self._average_traces = None
        self.model = PCA(2) if model is None else model
        self.compare_col = compare_col
        self.centerer = StandardScaler() if centerer is None else centerer
        self.traces = traces
        self._iscombined = self.traces.reset_index()[compare_col].nunique() > 1
        self.trajectories  # fit model


    def which_trajectories(self, method):
        if method == "Average":
            return self._get_trajectories_avg_projection
        elif method == "ByTrial":
            return self._get_trajectories_by_trial
    
    def which_average_trajectories(self, method):
        if method == "Average":
            return self._get_average_trajectories_avg_projection
        elif method == "ByTrial":
            return self._get_average_trajectories_by_trial


    def center(self, traces):
        return pd.DataFrame(self.centerer.fit_transform(traces.values), index=traces.index, columns=traces.columns)


    def _get_trajectories_avg_projection(self):
        self.model.fit(self.center(self.average_traces))
        transformed = self.model.transform(self.center(self.traces))
        return (
                pd.DataFrame(
                    transformed, index=self.traces.index, columns=["PC1", "PC2"]
                )
                .reset_index()
                .assign(prepost=lambda x: np.where(x["aligned"] < 0, "pre", "post"))
            )
    
    def _get_average_trajectories_avg_projection(self):
        transformed = self.model.transform(self.center(self.average_traces))
        average_trajectories = (
            pd.DataFrame(
                transformed, index=self.average_traces.index, columns=["PC1", "PC2"]
            )
            .reset_index()
            .assign(prepost=lambda x: np.where(x["aligned"] < 0, "pre", "post"))
        )
        average_trajectories = self._calculate_centroid_distance(
            average_trajectories
        )
        if self._iscombined:
            distances = self._get_pc_distance(trajectories)
            trajectories = trajectories.merge(distances)
        return trajectories
    

    def _get_trajectories_by_trial(self):
        self.model.fit(self.center(self.traces))
        transformed = self.model.transform(self.center(self.traces))
        return (
            pd.DataFrame(
                transformed, index=self.traces.index, columns=["PC1", "PC2"]
            )
            .reset_index()
            .assign(prepost=lambda x: np.where(x["aligned"] < 0, "pre", "post"))
        )
    
    def _get_average_trajectories_by_trial(self):
        trajectories = (
            self.trajectories.groupby([self.compare_col, "aligned"])[["PC1", "PC2"]]
            .mean()
            .reset_index()
            .assign(prepost=lambda x: np.where(x["aligned"] < 0, "pre", "post"))
        )
        trajectories = self._calculate_centroid_distance(trajectories)
        if self._iscombined:
            distances = self._get_pc_distance(trajectories)
            trajectories = trajectories.merge(distances)
        return trajectories


    def _get_pc_distance(self, average_trajectories):
        conditions = average_trajectories[self.compare_col].unique()
        a = average_trajectories.loc[lambda x: x[self.compare_col] == conditions[0]]
        b = average_trajectories.loc[lambda x: x[self.compare_col] == conditions[1]]
        if len(a) != len(b):
            if len(a) < len(b):
                b = b[b["aligned"].isin(a["aligned"].unique())]
            if len(b) < len(a):
                b = a[a["aligned"].isin(b["aligned"].unique())]
        distance_ts = np.linalg.norm(a[["PC1", "PC2"]].values - b[["PC1", "PC2"]].values, axis=1)
        return pd.DataFrame({"aligned": a["aligned"].values, "trajectory_distance": distance_ts})



    @property
    def scree(self):
        return get_scree(self.model)

    @property
    def loadings(self):
        return get_loadings(self.traces.columns, self.model)

    @property
    def average_traces(self):
        if self._average_traces is None:
            self._average_traces = (
                self.traces.reset_index().groupby([self.compare_col, "aligned"]).mean()
            ).drop("trial", axis=1)
        return self._average_traces
    
    @property
    def trajectories(self):
        if self._trajectories is None:
            self._trajectories = self._trajectories_getter()
        return self._trajectories
    
    @property
    def average_trajectories(self):
        if self._average_trajectories is None:
            self._average_trajectories = self._average_trajectories_getter()
        return self._average_trajectories
    

    @staticmethod
    def _get_difference_from_centroid(
        df_trial, returned_colname="distance_from_centroid"
    ):
        centroid = df_trial.loc[lambda x: x.aligned < 0][["PC1", "PC2"]].mean().values
        df_trial[returned_colname] = df_trial[["PC1", "PC2"]].apply(
            lambda x: distance.euclidean(x.values, centroid), axis=1
        )
        return df_trial

    def _calculate_centroid_distance(self, trajectories):
        trajectories = trajectories.groupby([self.compare_col]).apply(
            self._get_difference_from_centroid,
            returned_colname="distance_from_pre",
        )
        return trajectories




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
        self, average_trajectories, compare_col="session", cmaps=None, ax=None, markers=None,
    ):
        if ax is None:
            _, ax = plt.subplots()
        if cmaps is None:
            cmaps = ["seismic", "PRGn"]
        if markers is None:
            markers = ["o", "o"]
        for i, session in enumerate(average_trajectories[compare_col].unique()):
            df_session = average_trajectories.loc[lambda x: x[compare_col] == session]
            for prepost, marker in zip(["pre", "post"], markers):
                df_prepost = df_session.loc[lambda x: x.prepost == prepost]
                ax.scatter(
                    df_prepost["PC1"],
                    df_prepost["PC2"],
                    c=df_prepost["aligned"],
                    cmap=cmaps[i],
                    label=f"{session} {prepost}",
                    marker=marker,
                    vmin=average_trajectories["aligned"].min(),
                    vmax=average_trajectories["aligned"].max(),
                )
        ax.legend()
        PCM=ax.get_children()[2]
        plt.colorbar(PCM, ax=ax) 

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        if self.save:
            plt.savefig(self.save_dir / f"Trajectories - {self.name} - 2DScatter.svg")
        return ax

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
