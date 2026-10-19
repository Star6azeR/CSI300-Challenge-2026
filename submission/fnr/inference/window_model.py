"""Load frozen components and apply their fixed feature preprocessing."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from ..models import peer
from . import base as base_module


def rank(values):
    """Average percentile ranks, including the original tie convention."""
    return pd.Series(values).rank(pct=True).to_numpy()


class FrozenComponents:
    def __init__(self, model_dir):
        directory = Path(model_dir).resolve()
        manifest = json.loads((directory / "manifest.json").read_text())
        actual = {
            str(path.relative_to(directory))
            for path in directory.rglob("*")
            if path.is_file() and path.name != "manifest.json"
        }
        if actual != set(manifest["files_sha256"]):
            raise ValueError("Incomplete model manifest")
        for name, digest in manifest["files_sha256"].items():
            path = (directory / name).resolve()
            if (
                not path.is_relative_to(directory)
                or hashlib.sha256(path.read_bytes()).hexdigest() != digest
            ):
                raise ValueError("Model checksum mismatch")

        parent = directory / "frozen_base"
        self.base = base_module.InferenceModel(parent)
        self.pre = json.loads((parent / "factor/preprocessing.json").read_text())
        factor = parent / "factor/numpy_model"
        self.teacher = peer.core.load_numpy(factor / "teacher")
        self.p01 = peer.load_adapter(factor / "adapter")
        self.r1 = peer.load_adapter(factor / "adapter_r1")
        self.head = peer.load_adapter(directory / "head/adapter")

        factor_stats = parent / "statistical_correction"
        latent_stats = directory / "statistical"
        self.kcfg = json.loads((factor_stats / "config.json").read_text())
        self.scfg = json.loads((latent_stats / "config.json").read_text())
        self.kprior = np.load(factor_stats / "prior.npy", allow_pickle=False)
        self.kpen = np.load(factor_stats / "penalty.npy", allow_pickle=False)
        self.sprior = np.load(latent_stats / "prior.npy", allow_pickle=False)
        self.spen = np.load(latent_stats / "penalty.npy", allow_pickle=False)
        for array in [self.kprior, self.kpen, self.sprior, self.spen]:
            array.setflags(write=False)
        self.lookback = 120

    def standardize(self, frame):
        pre = self.pre
        values = frame[pre["features"]].to_numpy(np.float32)
        return np.clip(
            np.nan_to_num(
                (values - np.asarray(pre["mean"], np.float32))
                / np.asarray(pre["std"], np.float32),
                nan=0, posinf=0, neginf=0,
            ),
            -pre["clip"], pre["clip"],
        )
