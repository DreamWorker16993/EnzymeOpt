"""Stable, order-independent random-number stream derivation."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from numbers import Integral

import numpy as np
from numpy.random import Generator


SeedLabel = str | int


def derive_seed(master_seed: int, *labels: SeedLabel) -> int:
    """Derive a deterministic 128-bit seed from a master seed and labels."""

    _validate_seed(master_seed)
    digest = sha256()
    digest.update(b"enzymeopt.seed.v1\0")
    digest.update(int(master_seed).to_bytes(16, byteorder="big", signed=False))
    for label in labels:
        encoded = _encode_label(label)
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return int.from_bytes(digest.digest()[:16], byteorder="big", signed=False)


def make_rng(master_seed: int, *labels: SeedLabel) -> Generator:
    """Create a NumPy generator for one named random stream."""

    return np.random.Generator(np.random.PCG64(derive_seed(master_seed, *labels)))


@dataclass(frozen=True, slots=True)
class SeedManager:
    """Create reproducible random streams without mutable global RNG state."""

    master_seed: int

    def __post_init__(self) -> None:
        _validate_seed(self.master_seed)

    def seed_for(self, *labels: SeedLabel) -> int:
        return derive_seed(self.master_seed, *labels)

    def rng_for(self, *labels: SeedLabel) -> Generator:
        return make_rng(self.master_seed, *labels)


def _validate_seed(seed: object) -> None:
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0 or seed >= 2**128:
        raise ValueError("master_seed must be an integer in the range [0, 2**128)")


def _encode_label(label: SeedLabel) -> bytes:
    if isinstance(label, bool) or not isinstance(label, (str, Integral)):
        raise TypeError("seed labels must be strings or integers")
    if isinstance(label, str):
        return b"s\0" + label.encode("utf-8")
    return b"i\0" + str(int(label)).encode("ascii")
