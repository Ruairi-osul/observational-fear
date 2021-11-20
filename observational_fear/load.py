from pathlib import Path
import pandas as pd


def load_cells(data_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(data_dir / "cells.parquet.gzip")


def load_cell_mapper(data_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(data_dir / "cells_mapper.parquet.gzip")


def load_behaviour(data_dir: Path, session: str) -> pd.DataFrame:
    return pd.read_parquet(data_dir / f"{session}-behaviour.parquet.gzip")


def load_traces(data_dir: Path, session: str) -> pd.DataFrame:
    return pd.read_parquet(data_dir / f"{session}-traces.parquet.gzip")

