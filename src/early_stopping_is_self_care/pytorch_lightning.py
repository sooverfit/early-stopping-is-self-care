"""PyTorch Lightning (``pytorch_lightning``, the legacy package name) companion callback.

Drop it next to your ``EarlyStopping``; it copies that callback's monitor,
mode, patience and min_delta and reads the metric at the same hook. It only
reads ``trainer.callback_metrics`` and the early-stopping state, never sets
``trainer.should_stop``, and stores nothing in checkpoints.
"""

from __future__ import annotations

try:
    from pytorch_lightning.callbacks import Callback, EarlyStopping
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "early_stopping_is_self_care.pytorch_lightning needs pytorch_lightning. "
        "Try: pip install 'early-stopping-is-self-care[pytorch-lightning]'"
    ) from exc

from ._lightning_impl import LightningSelfCareMixin

__all__ = ["SelfCareCallback"]


class SelfCareCallback(LightningSelfCareMixin, Callback):
    """Prints one kind, nerdy roast when early stopping fires.

    Args:
        monitor, mode, patience, min_delta: what to watch. Unset values are
            copied from the trainer's ``EarlyStopping`` callback, if any.
        silent: compose the roast (``self.last_roast``) but print nothing.
            ``ESISC_SILENT=1`` does the same globally.
        seed: make roast selection deterministic.
        printer: where the roast goes, default ``print``. Pass e.g.
            ``logging.getLogger("lightning.pytorch").info``.
    """

    _EarlyStopping = EarlyStopping
