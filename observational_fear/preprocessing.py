from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
from binit.bin import which_bin_idx
from observational_fear import load


@dataclass
class Block:
    name: str
    length: int


def get_blocks(session: str,) -> Tuple[int, List[Block]]:
    def _get_day2_blocks() -> Tuple[int, List[Block]]:
        baseline = Block(name="Baseline", length=180)
        cs = Block(name="CS", length=28)
        us = Block(name="US", length=2)
        iti = Block(name="ITI", length=30)
        post = Block(name="Post", length=180)
        blocks = [baseline] + ([cs, us, iti] * 30) + [post]
        return 3, blocks

    def _get_day4_blocks() -> Tuple[int, List[Block]]:
        baseline = Block("Baseline", length=208)
        us = Block("US", length=2)
        iti = Block("ITI", length=58)
        post = Block("Post", length=180)
        blocks = [baseline] + ([us, iti] * 30) + [post]
        return 2, blocks

    if session == "day2":
        f = _get_day2_blocks
    elif session == "day4":
        f = _get_day4_blocks
    else:
        raise ValueError("Unknown Session")
    num_blocks, blocks = f()
    return num_blocks, blocks


def get_blocknames_trialnumber(
    time: np.ndarray, session: str,
) -> Tuple[np.ndarray, np.ndarray]:
    @np.vectorize
    def _get_trail(block_number: float, num_blocks_per_trial: int):
        if block_number < 1 or block_number > 90:
            return np.nan
        trial_zero_index = (block_number - 1) // num_blocks_per_trial
        return trial_zero_index + 1

    num_blocks_per_trial, blocks = get_blocks(session)
    block_starts = np.cumsum([0] + [block.length for block in blocks])
    block_number = which_bin_idx(time, bin_edges=block_starts)
    trial_number = _get_trail(block_number, num_blocks_per_trial)
    block_names = np.array(list(blocks[idx].name for idx in block_number.astype(int)))
    return block_names, trial_number


def get_coreg_cells(data_dir, sessions):
    return (
        load.load_cell_mapper(data_dir)
        .assign(dummy=1)
        .pivot(index="new_id", columns="session",values="dummy")
        .fillna(0)
        [sessions]
        .sum(axis=1)
        .loc[lambda x: x==len(sessions)]
        .index
        .values
        .astype(str)
    )