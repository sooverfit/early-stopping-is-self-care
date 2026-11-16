"""Keras companion callback (Keras 3, or tf.keras on TensorFlow 2.x).

It only reads ``logs`` and, if you pass one, your ``EarlyStopping`` instance.
It never sets ``model.stop_training``, never edits ``logs`` and never touches
weights, so training behaves exactly as it would without it.
"""

from __future__ import annotations

from typing import Any, Optional

from .core import Companion, Printer

try:  # Keras 3 (any backend)
    import keras as _keras

    _Callback = _keras.callbacks.Callback
except ImportError:  # pragma: no cover - TF 2.x with bundled tf.keras
    try:
        from tensorflow import keras as _keras  # type: ignore[no-redef]

        _Callback = _keras.callbacks.Callback
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "early_stopping_is_self_care.keras needs Keras. "
            "Try: pip install 'early-stopping-is-self-care[keras]' plus a backend "
            "(tensorflow, torch or jax)."
        ) from exc

__all__ = ["SelfCareCallback"]

_UNSET: Any = object()


class SelfCareCallback(_Callback):
    """Prints one kind, nerdy roast when early stopping fires.

    Args:
        monitor, mode, patience, min_delta: what to watch. If ``early_stopping``
            is given, anything left unset is copied from it.
        early_stopping: your ``keras.callbacks.EarlyStopping`` instance
            (recommended). It is only ever read.
        silent: compose the roast (``self.last_roast``) but print nothing.
            ``ESISC_SILENT=1`` does the same globally.
        seed: make roast selection deterministic.
        printer: where the roast goes, default ``print``. Pass e.g.
            ``logging.getLogger(__name__).info``.
    """

    def __init__(
        self,
        monitor: Optional[str] = None,
        mode: Optional[str] = None,
        patience: Optional[int] = None,
        min_delta: Optional[float] = None,
        *,
        early_stopping: Any = None,
        silent: bool = False,
        seed: Optional[int] = None,
        printer: Optional[Printer] = None,
    ):
        super().__init__()
        es = early_stopping
        self.early_stopping = es
        monitor = monitor or getattr(es, "monitor", None) or "val_loss"
        mode = mode or getattr(es, "mode", None) or "auto"
        if patience is None:
            patience = getattr(es, "patience", 0) or 0
        if min_delta is None:
            min_delta = abs(getattr(es, "min_delta", 0.0) or 0.0)
        self._companion = Companion(monitor, mode, patience, min_delta, silent, seed, printer)

    @property
    def last_roast(self) -> Optional[str]:
        return self._companion.last_roast

    @property
    def silent(self) -> bool:
        return self._companion.silent

    @silent.setter
    def silent(self, value: bool) -> None:
        self._companion.silent = bool(value)

    def on_train_begin(self, logs=None):
        self._companion.begin()

    def on_epoch_end(self, epoch, logs=None):
        value = (logs or {}).get(self._companion.monitor)
        if value is not None:
            self._companion.observe(int(epoch) + 1, value)

    def on_train_end(self, logs=None):
        es = self.early_stopping
        if es is not None and getattr(es, "stopped_epoch", 0):
            best_epoch = getattr(es, "best_epoch", None)
            best = getattr(es, "best", None)
            try:
                best = float(best) if best is not None else None
            except (TypeError, ValueError):
                best = None
            self._companion.finish(
                stopped=True,
                best_epoch=(int(best_epoch) + 1) if best_epoch is not None else None,
                last_epoch=int(es.stopped_epoch) + 1,
                best=best,
            )
            return
        model = getattr(self, "model", None)
        stopped = bool(getattr(model, "stop_training", False))
        self._companion.finish(stopped=stopped)
