"""Shared implementation for the ``lightning`` and ``pytorch_lightning`` callbacks."""

from __future__ import annotations

from typing import Any, Optional

from .core import Companion, Printer


class LightningSelfCareMixin:
    """Mixed into the framework's ``Callback``; see ``SelfCareCallback``."""

    _EarlyStopping: Any = None  # set by the concrete subclass

    def __init__(
        self,
        monitor: Optional[str] = None,
        mode: Optional[str] = None,
        patience: Optional[int] = None,
        min_delta: Optional[float] = None,
        *,
        silent: bool = False,
        seed: Optional[int] = None,
        printer: Optional[Printer] = None,
    ):
        super().__init__()
        self._explicit = {"monitor": monitor, "mode": mode, "patience": patience,
                          "min_delta": min_delta}
        self._companion = Companion(monitor or "val_loss", mode or "min", patience or 0,
                                    min_delta or 0.0, silent, seed, printer)
        self._check_on_train_epoch_end = True
        self._paired: Any = None

    @property
    def last_roast(self) -> Optional[str]:
        return self._companion.last_roast

    @property
    def silent(self) -> bool:
        return self._companion.silent

    @silent.setter
    def silent(self, value: bool) -> None:
        self._companion.silent = bool(value)

    def _find_early_stopping(self, trainer) -> Any:
        for cb in getattr(trainer, "callbacks", []) or []:
            if self._EarlyStopping is not None and isinstance(cb, self._EarlyStopping):
                return cb
        return None

    def setup(self, trainer, pl_module, stage: str) -> None:
        if stage != "fit":
            return
        es = self._find_early_stopping(trainer)
        self._paired = es
        ex = self._explicit
        monitor = ex["monitor"] or getattr(es, "monitor", None) or "val_loss"
        mode = ex["mode"] or getattr(es, "mode", None) or "min"
        patience = ex["patience"] if ex["patience"] is not None else getattr(es, "patience", 0)
        min_delta = ex["min_delta"] if ex["min_delta"] is not None else abs(
            float(getattr(es, "min_delta", 0.0)))
        self._companion.configure(monitor, mode, patience, min_delta)
        # Read the metric at the same hook EarlyStopping checks it.
        es_flag = getattr(es, "_check_on_train_epoch_end", None)
        if es_flag is not None:
            self._check_on_train_epoch_end = bool(es_flag)
        else:
            self._check_on_train_epoch_end = (
                trainer.val_check_interval == 1.0 and trainer.check_val_every_n_epoch == 1
            )

    def on_fit_start(self, trainer, pl_module) -> None:
        self._companion.begin()

    def _record(self, trainer) -> None:
        if getattr(trainer, "sanity_checking", False):
            return
        value = trainer.callback_metrics.get(self._companion.monitor)
        if value is None:
            return
        if hasattr(value, "detach"):
            value = value.detach().cpu().item()
        self._companion.observe(int(trainer.current_epoch) + 1, value)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if self._check_on_train_epoch_end:
            self._record(trainer)

    def on_validation_end(self, trainer, pl_module) -> None:
        if not self._check_on_train_epoch_end:
            self._record(trainer)

    def on_train_end(self, trainer, pl_module) -> None:
        es = self._paired
        stopped_epoch = int(getattr(es, "stopped_epoch", 0) or 0) if es is not None else 0
        stopped = bool(stopped_epoch) or bool(getattr(trainer, "should_stop", False))
        last_epoch = stopped_epoch + 1 if stopped_epoch else None
        if getattr(trainer, "is_global_zero", True):
            self._companion.finish(stopped=stopped, last_epoch=last_epoch)
        else:
            self._companion._done = True
