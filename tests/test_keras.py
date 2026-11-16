"""Behaviour-equivalence proof for the Keras callback.

Each test trains the same seeded tiny model twice on CPU, once with
EarlyStopping alone and once with EarlyStopping + SelfCareCallback, and
asserts the runs are bit-for-bit identical.
"""

import pytest

np = pytest.importorskip("numpy")
keras = pytest.importorskip("keras")

from early_stopping_is_self_care.core import PREFIX  # noqa: E402
from early_stopping_is_self_care.keras import SelfCareCallback  # noqa: E402

MAX_EPOCHS = 200


def _data():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(8, 1))
    x = rng.normal(size=(48, 8)).astype("float32")
    y = (x @ w + rng.normal(scale=1.5, size=(48, 1))).astype("float32")
    xv = rng.normal(size=(64, 8)).astype("float32")
    yv = (xv @ w + rng.normal(scale=1.5, size=(64, 1))).astype("float32")
    return x, y, xv, yv


def _run(extra_callbacks=(), with_es=True, order="after", patience=4):
    keras.utils.set_random_seed(0)
    x, y, xv, yv = _data()
    model = keras.Sequential([
        keras.Input((8,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(0.01), loss="mse")
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                       restore_best_weights=True) if with_es else None
    extras = [cb(es) if callable(cb) else cb for cb in extra_callbacks]
    cbs = ([es] if es else [])
    cbs = cbs + extras if order == "after" else extras + cbs
    hist = model.fit(x, y, validation_data=(xv, yv), epochs=MAX_EPOCHS, batch_size=16,
                     shuffle=True, verbose=0, callbacks=cbs)
    return {
        "epochs_run": len(hist.history["loss"]),
        "history": hist.history,
        "stopped_epoch": es.stopped_epoch if es else None,
        "best_epoch": es.best_epoch if es else None,
        "weights": model.get_weights(),
        "pred": np.asarray(model.predict(xv, verbose=0)),
        "es": es,
        "extras": extras,
    }


def _assert_identical(a, b):
    assert a["epochs_run"] == b["epochs_run"]
    assert a["stopped_epoch"] == b["stopped_epoch"]
    assert a["best_epoch"] == b["best_epoch"]
    assert a["history"] == b["history"]
    assert len(a["weights"]) == len(b["weights"])
    for wa, wb in zip(a["weights"], b["weights"]):
        assert np.array_equal(wa, wb)
    assert np.array_equal(a["pred"], b["pred"])


def test_control_seeded_runs_are_reproducible():
    a, b = _run(), _run()
    assert 0 < a["stopped_epoch"] < MAX_EPOCHS - 1, "EarlyStopping must actually fire"
    _assert_identical(a, b)


@pytest.mark.parametrize("order", ["after", "before"])
def test_behaviour_unchanged_with_early_stopping(order, capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base = _run()
    capsys.readouterr()
    comp = _run([lambda es: SelfCareCallback(early_stopping=es, seed=0)], order=order)
    out = capsys.readouterr().out
    assert base["stopped_epoch"] > 0
    _assert_identical(base, comp)
    assert out.count(PREFIX) == 1
    roast = comp["extras"][0].last_roast
    assert roast and roast in out
    assert list(tmp_path.iterdir()) == []  # nothing written


def test_behaviour_unchanged_bare_callback_inferred_from_stop_training(capsys):
    base = _run()
    capsys.readouterr()
    cb = SelfCareCallback(monitor="val_loss", patience=4, seed=0)
    comp = _run([cb])
    _assert_identical(base, comp)
    assert capsys.readouterr().out.count(PREFIX) == 1
    assert cb.last_roast
    # Its own tracker agrees with EarlyStopping about the best epoch (1-based).
    assert cb._companion.tracker.best_epoch == base["best_epoch"] + 1


def test_behaviour_unchanged_without_early_stopping(capsys):
    global MAX_EPOCHS
    old, MAX_EPOCHS = MAX_EPOCHS, 25
    try:
        base = _run(with_es=False)
        capsys.readouterr()
        cb = SelfCareCallback(seed=0)
        comp = _run([cb], with_es=False)
    finally:
        MAX_EPOCHS = old
    _assert_identical(base, comp)
    assert comp["epochs_run"] == 25
    out = capsys.readouterr().out
    # The run passes its best epoch, so it gets a (passed-peak) roast, once.
    assert out.count(PREFIX) == 1


def test_silent_mode_prints_nothing(capsys):
    res = _run([lambda es: SelfCareCallback(early_stopping=es, seed=0, silent=True)])
    assert capsys.readouterr().out == ""
    assert res["stopped_epoch"] > 0
    assert res["extras"][0].last_roast  # composed, just not printed


def test_seeded_roast_is_deterministic(capsys):
    a = _run([lambda es: SelfCareCallback(early_stopping=es, seed=42)])["extras"][0].last_roast
    b = _run([lambda es: SelfCareCallback(early_stopping=es, seed=42)])["extras"][0].last_roast
    assert a == b


def test_printer_hook_receives_roast():
    got = []
    _run([lambda es: SelfCareCallback(early_stopping=es, seed=0, printer=got.append)])
    assert len(got) == 1 and got[0].startswith(PREFIX)
