# early-stopping-is-self-care

[![Is my run overfitting? (Hugging Face Space)](https://img.shields.io/badge/%F0%9F%A4%97%20Space-is--my--run--overfitting-yellow)](https://huggingface.co/spaces/sooverfit/is-my-run-overfitting)
[![overfit: probably](https://sooverfit-clicks.utopian-cabbage.workers.dev/badge/overfit.svg)](https://sooverfit-clicks.utopian-cabbage.workers.dev/r/20261117-badge)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

A drop-in companion to your early-stopping callback for **Keras** and
**PyTorch Lightning**. When early stopping fires, it prints one short, kind,
ML-literate roast of your run. That is all it does.

(Real output from the Keras test model with `seed=31`.)

```text
Epoch 10: early stopping
Restoring model weights from the end of the best epoch: 6.
[early-stopping-is-self-care] Your val_loss peaked at epoch 6. Everything after was just memorising the training set's birthday.
```

- Zero dependencies for the core. Frameworks are optional extras.
- No telemetry, no network calls, no file writes.
- It never changes training or early-stopping behaviour (see below; there are tests).

## Install

```bash
pip install "early-stopping-is-self-care[keras]"       # Keras 3 (bring a backend: tensorflow, torch or jax)
pip install "early-stopping-is-self-care[lightning]"   # lightning.pytorch
pip install "early-stopping-is-self-care[pytorch-lightning]"  # legacy pytorch_lightning package
pip install early-stopping-is-self-care                # just the core + diagnose()
```

Requires Python 3.9+.

## Quick start: Keras

```python
import keras
from early_stopping_is_self_care.keras import SelfCareCallback

early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

model.fit(
    x, y,
    validation_data=(x_val, y_val),
    epochs=200,
    callbacks=[early_stop, SelfCareCallback(early_stopping=early_stop)],
)
```

Pass your `EarlyStopping` instance so the roast uses its exact best and stopped
epochs. Without it, `SelfCareCallback(monitor="val_loss")` watches the metric
itself with the same `min_delta` rules and notices when training was stopped.
Works with Keras 3 on any backend, and with `tf.keras` on TensorFlow 2.x.

## Quick start: PyTorch Lightning

```python
import lightning.pytorch as L
from lightning.pytorch.callbacks import EarlyStopping
from early_stopping_is_self_care.lightning import SelfCareCallback

trainer = L.Trainer(
    max_epochs=200,
    callbacks=[EarlyStopping(monitor="val_loss", patience=5), SelfCareCallback()],
)
trainer.fit(model, train_loader, val_loader)
```

`SelfCareCallback()` finds the trainer's `EarlyStopping` and copies its
`monitor`, `mode`, `patience` and `min_delta`, and reads the metric at the same
hook. On multi-device runs it only speaks on global rank zero. Using the
older package name? `from early_stopping_is_self_care.pytorch_lightning import SelfCareCallback`.

## Options

Both callbacks take the same keyword arguments:

| Argument | Default | What it does |
| --- | --- | --- |
| `monitor`, `mode`, `patience`, `min_delta` | copied from `EarlyStopping`, else `val_loss` | What to watch |
| `silent` | `False` | Compose the roast (available as `.last_roast`) but print nothing |
| `seed` | `None` | Deterministic roast selection |
| `printer` | `print` | Where the roast goes, e.g. `logging.getLogger(__name__).info` |

Set `ESISC_SILENT=1` to silence every roast without touching code (handy in CI).

The roast prints once per `fit()`, when early stopping fired, or when the run
ended after its best epoch. A run that ends on its best epoch gets nothing.
Epochs in roasts are counted from 1.

## It changes nothing

This callback is read-only by design:

- It never sets `model.stop_training` (Keras) or `trainer.should_stop` (Lightning).
- It never edits `logs`, metrics, weights, the optimiser or the data.
- It stores no state in Lightning checkpoints (`state_dict()` is empty).
- It imports no networking code and opens no files.

The test suite proves it on CPU. For both Keras and Lightning it trains the
same seeded tiny model twice, once with `EarlyStopping` alone and once with
`EarlyStopping` plus `SelfCareCallback` (in either callback order), and asserts
the two runs are bit-for-bit identical: the same stop epoch, best epoch, history
or callback metrics, and final weights. Sockets are patched to fail during every
test.

## Bonus: `diagnose()` with plain lists

```python
from early_stopping_is_self_care import diagnose

print(diagnose(train_losses, val_losses, patience=5, seed=0))
```

```text
Verdict: overfitting
Best val at epoch 20 (0.4667); early stopping (patience=5) would fire at epoch 25.
Generalisation gap (val - train): 0.2515 at best, 0.505 at the end.
- 30 epochs; train 1.18 -> 0.1249, val 1.18 -> 0.6299.
- Val rose 22.9% of scale after its minimum; gap is widening.
- Val zigzag rate 4%.
Early stopping noticed what validation had been hinting since epoch 20. Healthy communication, finally.
```

Pure Python heuristics, no numpy. For plots and a shareable card, try the
[overfitting checker Space](https://huggingface.co/spaces/sooverfit/is-my-run-overfitting).
Or just `python -m early_stopping_is_self_care` for a roast on demand.

## Satire note

The jokes are jokes. Early stopping is a real and useful regulariser; this
package only makes fun of the epochs after your best one. The companion satire
paper, *Early Stopping Is Self-Care*, is here:
[early-stopping-is-self-care.pdf](https://huggingface.co/spaces/sooverfit/is-my-run-overfitting/resolve/main/early-stopping-is-self-care.pdf).

## Add the badge to your repo

A joke badge. It reads **overfit | probably** and claims to measure nothing.

```markdown
[![overfit: probably](https://sooverfit-clicks.utopian-cabbage.workers.dev/badge/overfit.svg)](https://sooverfit-clicks.utopian-cabbage.workers.dev/r/20261117-badge)
```

## Made by So Overfit

Merch (our shop): [Early stopping is self-care tee](https://sooverfit-clicks.utopian-cabbage.workers.dev/r/20261116-esisc-github)

MIT licensed. Contributions welcome, see [CONTRIBUTING.md](CONTRIBUTING.md).
