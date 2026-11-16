"""Framework-agnostic bits: metric tracking and roast selection.

Nothing here touches training. It only reads numbers it is handed.
"""

from __future__ import annotations

import math
import os
import random
from typing import Callable, Optional

from . import roasts as _roasts

PREFIX = "[early-stopping-is-self-care] "

_MAX_HINTS = ("acc", "auc", "f1", "precision", "recall", "iou", "dice", "r2", "fbeta")

Printer = Callable[[str], None]


def resolve_mode(monitor: str, mode: str = "auto") -> str:
    """Return ``"min"`` or ``"max"``. ``auto`` guesses from the metric name."""
    if mode in ("min", "max"):
        return mode
    if mode != "auto":
        raise ValueError(f"mode must be 'min', 'max' or 'auto', not {mode!r}")
    name = monitor.lower()
    if "loss" in name or "error" in name:
        return "min"
    return "max" if any(h in name for h in _MAX_HINTS) else "min"


def env_silenced() -> bool:
    """``ESISC_SILENT=1`` silences every roast without touching code."""
    return os.environ.get("ESISC_SILENT", "").strip().lower() in ("1", "true", "yes", "on")


class PlateauTracker:
    """Tracks the best value of a metric with EarlyStopping-style min_delta rules.

    An epoch counts as an improvement when ``value < best - min_delta`` (min mode)
    or ``value > best + min_delta`` (max mode), matching Keras and Lightning.
    Epochs are 1-based.
    """

    def __init__(self, monitor: str = "val_loss", mode: str = "auto", min_delta: float = 0.0):
        self.monitor = monitor
        self.mode = resolve_mode(monitor, mode)
        self.min_delta = abs(float(min_delta))
        self.reset()

    def reset(self) -> None:
        self.best: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.last_epoch: Optional[int] = None

    def is_improvement(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def update(self, epoch: int, value: float) -> None:
        value = float(value)
        if math.isnan(value):
            self.last_epoch = epoch
            return
        if self.is_improvement(value):
            self.best = value
            self.best_epoch = epoch
        self.last_epoch = epoch

    @property
    def epochs_since_best(self) -> int:
        if self.best_epoch is None or self.last_epoch is None:
            return 0
        return self.last_epoch - self.best_epoch


def _plural(n: int) -> str:
    return f"{n} epoch" if n == 1 else f"{n} epochs"


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4g}"


def compose_roast(
    *,
    stopped: bool,
    best_epoch: int,
    last_epoch: int,
    monitor: str = "val_loss",
    best: Optional[float] = None,
    patience: int = 0,
    rng: Optional[random.Random] = None,
) -> str:
    """Pick and fill one roast. Pass a seeded ``random.Random`` for determinism."""
    rng = rng or random.Random()
    extra = max(0, last_epoch - best_epoch)
    if extra < 1:
        return _roasts.FALLBACK
    pool = _roasts.GENERAL + (_roasts.STOPPED if stopped else _roasts.PASSED_PEAK)
    template = rng.choice(pool)
    return template.format(
        best_epoch=best_epoch,
        last_epoch=last_epoch,
        extra_epochs=_plural(extra),
        monitor=monitor,
        best=_fmt(best),
        patience=patience,
    )


def roast(seed: Optional[int] = None, best_epoch: int = 22, last_epoch: int = 27,
          monitor: str = "val_loss", stopped: bool = True) -> str:
    """A roast without any training. Handy for slides, stand-ups and tees."""
    return compose_roast(stopped=stopped, best_epoch=best_epoch, last_epoch=last_epoch,
                         monitor=monitor, rng=random.Random(seed), patience=5)


class Companion:
    """Shared state machine used by the Keras and Lightning callbacks."""

    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "auto",
        patience: int = 0,
        min_delta: float = 0.0,
        silent: bool = False,
        seed: Optional[int] = None,
        printer: Optional[Printer] = None,
    ):
        self.monitor = monitor
        self.patience = int(patience)
        self.silent = bool(silent)
        self.seed = seed
        self.printer: Printer = printer or print
        self.tracker = PlateauTracker(monitor, mode, min_delta)
        self.last_roast: Optional[str] = None
        self._done = False
        self._rng = random.Random(seed)

    def configure(self, monitor: str, mode: str, patience: int, min_delta: float) -> None:
        self.monitor = monitor
        self.patience = int(patience)
        self.tracker = PlateauTracker(monitor, mode, min_delta)

    def begin(self) -> None:
        self.tracker.reset()
        self.last_roast = None
        self._done = False
        self._rng = random.Random(self.seed)

    def observe(self, epoch: int, value) -> None:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return
        self.tracker.update(epoch, v)

    def finish(
        self,
        stopped: bool,
        best_epoch: Optional[int] = None,
        last_epoch: Optional[int] = None,
        best: Optional[float] = None,
    ) -> Optional[str]:
        """Compose (and, unless silenced, emit) at most one roast per run."""
        if self._done:
            return self.last_roast
        self._done = True
        best_epoch = best_epoch if best_epoch is not None else self.tracker.best_epoch
        last_epoch = last_epoch if last_epoch is not None else self.tracker.last_epoch
        best = best if best is not None else self.tracker.best
        if best_epoch is None or last_epoch is None:
            return None
        if not stopped and last_epoch <= best_epoch:
            return None  # ended on its best epoch: nothing to roast
        self.last_roast = compose_roast(
            stopped=stopped,
            best_epoch=best_epoch,
            last_epoch=last_epoch,
            monitor=self.monitor,
            best=best,
            patience=self.patience,
            rng=self._rng,
        )
        if not (self.silent or env_silenced()):
            self.printer(PREFIX + self.last_roast)
        return self.last_roast
