"""Plain-Python learning-curve diagnosis. No numpy, no frameworks.

A lighter cousin of the heuristics in the "Is my run overfitting?" Space.
These are heuristics on the numbers you pass in, not a proof.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .core import compose_roast

VERDICTS = ("healthy", "underfitting", "overfitting", "noisy")


@dataclass
class Diagnosis:
    verdict: str
    best_epoch: int
    best_val: float
    gap_at_best: float
    gap_at_end: float
    early_stop_epoch: Optional[int]
    patience: int
    notes: List[str] = field(default_factory=list)
    roast: str = ""

    def __str__(self) -> str:
        stop = (
            f"early stopping (patience={self.patience}) would fire at epoch {self.early_stop_epoch}"
            if self.early_stop_epoch is not None
            else f"early stopping (patience={self.patience}) would not have fired"
        )
        lines = [
            f"Verdict: {self.verdict}",
            f"Best val at epoch {self.best_epoch} ({self.best_val:.4g}); {stop}.",
            f"Generalisation gap (val - train): {self.gap_at_best:.4g} at best, "
            f"{self.gap_at_end:.4g} at the end.",
        ]
        lines += [f"- {n}" for n in self.notes]
        if self.roast:
            lines.append(self.roast)
        return "\n".join(lines)


def _slope(ys: Sequence[float]) -> float:
    n = len(ys)
    if n < 2:
        return 0.0
    xm = (n - 1) / 2.0
    ym = sum(ys) / n
    den = sum((i - xm) ** 2 for i in range(n))
    return sum((i - xm) * (y - ym) for i, y in enumerate(ys)) / den


def _zigzag(ys: Sequence[float]) -> float:
    d = [b - a for a, b in zip(ys, ys[1:]) if abs(b - a) > 1e-15]
    if len(d) < 2:
        return 0.0
    flips = sum(1 for a, b in zip(d, d[1:]) if a * b < 0)
    return flips / (len(d) - 1)


def early_stop_epoch(val: Sequence[float], patience: int, min_delta: float = 0.0) -> Optional[int]:
    """1-based epoch at which a min-mode early stopper would fire, else None."""
    best = math.inf
    wait = 0
    for i, v in enumerate(val):
        if v < best - abs(min_delta):
            best, wait = v, 0
        else:
            wait += 1
            if wait >= max(patience, 1):
                return i + 1
    return None


def diagnose(
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    patience: int = 5,
    min_delta: float = 0.0,
    seed: Optional[int] = None,
) -> Diagnosis:
    """Diagnose paired per-epoch train/val losses (lower is better)."""
    train = [float(x) for x in train_losses]
    val = [float(x) for x in val_losses]
    if len(train) != len(val):
        raise ValueError(f"train and val must be the same length ({len(train)} vs {len(val)})")
    pairs = [(t, v) for t, v in zip(train, val) if math.isfinite(t) and math.isfinite(v)]
    if len(pairs) < 2:
        raise ValueError("need at least two finite (train, val) pairs")
    train = [p[0] for p in pairs]
    val = [p[1] for p in pairs]
    n = len(val)

    best_i = min(range(n), key=lambda i: val[i])
    best_val = val[best_i]
    gaps = [v - t for t, v in zip(train, val)]
    scale = max(abs(best_val), abs(val[0] - best_val), sorted(abs(v) for v in val)[n // 2], 1e-12)
    rel_rise = (val[-1] - best_val) / scale
    tail = max(3, n // 5)
    train_tail = _slope(train[-tail:]) * (tail - 1) / scale
    val_tail = _slope(val[-tail:]) * (tail - 1) / scale
    k = max(2, n // 5)
    gap_delta = sum(gaps[-k:]) / k - sum(gaps[:k]) / k
    widening = gap_delta > 0.05 * scale
    zz = _zigzag(val)
    stop = early_stop_epoch(val, patience, min_delta)

    notes = [
        f"{n} epochs; train {train[0]:.4g} -> {train[-1]:.4g}, val {val[0]:.4g} -> {val[-1]:.4g}.",
        f"Val rose {rel_rise * 100:.1f}% of scale after its minimum; gap is "
        f"{'widening' if widening else 'not widening'}.",
        f"Val zigzag rate {zz:.0%}.",
    ]

    still_dropping = train_tail < -0.04 and val_tail < -0.02
    rebounded = rel_rise > 0.05 and best_i < n - max(1, patience // 2)
    if n >= 8 and zz >= 0.45 and not (rebounded and widening):
        verdict = "noisy"
    elif rebounded and (widening or train_tail <= 0.01):
        verdict = "overfitting"
    elif still_dropping and not rebounded:
        verdict = "underfitting"
    else:
        verdict = "healthy"
    if n < 5:
        notes.append("Very short run: treat the verdict as a vibe check.")

    roast = ""
    if best_i < n - 1:
        roast = compose_roast(
            stopped=stop is not None,
            best_epoch=best_i + 1,
            last_epoch=stop if stop is not None else n,
            monitor="val loss",
            best=best_val,
            patience=patience,
            rng=random.Random(seed),
        )
    return Diagnosis(
        verdict=verdict,
        best_epoch=best_i + 1,
        best_val=best_val,
        gap_at_best=gaps[best_i],
        gap_at_end=gaps[-1],
        early_stop_epoch=stop,
        patience=patience,
        notes=notes,
        roast=roast,
    )
