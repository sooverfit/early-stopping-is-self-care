# Contributing

Thanks for wanting to help a callback that does nothing do nothing better.

## Ground rules

- **It must change nothing.** No change may alter training, early stopping,
  metrics, weights or checkpoints. The equivalence tests in `tests/` must pass.
- **No telemetry, no network calls, no file writes.** The core keeps zero
  dependencies.
- **Jokes** go in `src/early_stopping_is_self_care/roasts.py`. They must be
  original (no quoting real people), kind (punch at the curve, never at the
  person), and ML-literate. Only put a joke in `STOPPED` if it claims
  early stopping fired; `GENERAL` jokes must be true either way.

## Dev setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -e ".[keras,lightning,test]"
KERAS_BACKEND=torch pytest
```

## Pull requests

One topic per PR, with tests for behaviour changes. Be nice in reviews; the
model already gets roasted enough.
