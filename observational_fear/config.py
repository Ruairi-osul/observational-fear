from dataclasses import dataclass, field
from locale import normalize
from typing import Optional
from pathlib import Path

@dataclass
class Config:
    data_dir: Optional[Path] = None
    session: Optional[str] = None
    t_before: Optional[float] = None
    t_after: Optional[float] = None
    normalize: Optional[bool] = False
    coreg_only: bool = False
    sampling_interval: str = "100ms"

@dataclass
class BlockConfig(Config):
    block: Optional[str] = None
    is_freeze: bool = field(init=False, default=False)


@dataclass
class FreezeConfig(Config):
    start_stop: Optional[str] = None
    role: str = "obs"
    min_freeze_diff: int = 11
    is_freeze: bool = field(init=False, default=True)
