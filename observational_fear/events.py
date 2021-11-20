import numpy as np
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


def get_freeze_starts(freezes: np.ndarray) -> np.ndarray:
    return np.where(np.diff(freezes == 1, 1, 0))
