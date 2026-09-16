from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch


@dataclass
class Episode:
    obs: torch.FloatTensor
    act: torch.LongTensor
    rew: torch.FloatTensor
    end: torch.ByteTensor
    trunc: torch.ByteTensor
    info: Dict[str, Any]

    def __len__(self) -> int:
        return self.obs.size(0)

    def __add__(self, other: Episode) -> Episode:
        assert self.dead.sum() == 0
        d = {k: torch.cat((v, other.__dict__[k]), dim=0) for k, v in self.__dict__.items() if k != "info"}
        return Episode(**d, info=merge_info(self.info, other.info))

    def to(self, device) -> Episode:
        return Episode(**{k: v.to(device) if k != "info" else v for k, v in self.__dict__.items()})

    @property
    def dead(self) -> torch.ByteTensor:
        return (self.end + self.trunc).clip(max=1)

    def compute_metrics(self) -> Dict[str, Any]:
        return {"length": len(self), "return": self.rew.sum().item()}

    @classmethod
    def load(cls, path: Path, map_location: Optional[torch.device] = None) -> Episode:
        raw = torch.load(Path(path), map_location=map_location)
        d = {}
        for k, v in raw.items():
            if k == "obs" and isinstance(v, torch.Tensor) and v.dtype == torch.uint8:
                d[k] = v.float().div(255.0).mul(2.0).sub(1.0)
            elif k == "info" and isinstance(v, dict):
                info_d = {}
                for ik, iv in v.items():
                    if ik == "final_observation" and isinstance(iv, torch.Tensor) and iv.dtype == torch.uint8:
                        info_d[ik] = iv.float().div(255.0).mul(2.0).sub(1.0)
                    else:
                        info_d[ik] = iv
                d[k] = info_d
            else:
                d[k] = v
        return cls(**d)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = {}
        for k, v in self.__dict__.items():
            if k == "obs" and isinstance(v, torch.Tensor) and v.dtype != torch.uint8:
                d[k] = v.add(1.0).div(2.0).mul(255.0).byte()
            elif k == "info" and isinstance(v, dict):
                info_d = {}
                for ik, iv in v.items():
                    if ik == "final_observation" and isinstance(iv, torch.Tensor) and iv.dtype != torch.uint8:
                        info_d[ik] = iv.add(1.0).div(2.0).mul(255.0).byte()
                    else:
                        info_d[ik] = iv
                d[k] = info_d
            else:
                d[k] = v
        torch.save(d, path.with_suffix(".tmp"))
        path.with_suffix(".tmp").rename(path)


def merge_info(info_a, info_b):
    keys_a = set(info_a)
    keys_b = set(info_b)
    intersection = keys_a & keys_b
    info = {
        **{k: info_a[k] for k in keys_a if k not in intersection},
        **{k: info_b[k] for k in keys_b if k not in intersection},
        **{k: torch.cat((info_a[k], info_b[k]), dim=0) for k in intersection},
    }
    return info
