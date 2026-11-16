"""Behaviour-equivalence proof for the Lightning callback.

Same seeded tiny model trained twice on CPU: EarlyStopping alone vs
EarlyStopping + SelfCareCallback. Stop epoch, metrics and final weights must
be identical.
"""

import pytest

torch = pytest.importorskip("torch")
L = pytest.importorskip("lightning.pytorch")

from lightning.pytorch.callbacks import EarlyStopping  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from early_stopping_is_self_care.core import PREFIX  # noqa: E402
from early_stopping_is_self_care.lightning import SelfCareCallback  # noqa: E402

MAX_EPOCHS = 200


class Tiny(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(8, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, _):
        x, y = batch
        loss = torch.nn.functional.mse_loss(self(x), y)
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        self.log("val_loss", torch.nn.functional.mse_loss(self(x), y), on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)


def _loaders():
    g = torch.Generator().manual_seed(0)
    w = torch.randn(8, 1, generator=g)
    x = torch.randn(48, 8, generator=g)
    y = x @ w + 1.5 * torch.randn(48, 1, generator=g)
    xv = torch.randn(64, 8, generator=g)
    yv = xv @ w + 1.5 * torch.randn(64, 1, generator=g)
    return (DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True),
            DataLoader(TensorDataset(xv, yv), batch_size=64))


def _run(root, extra=(), with_es=True, order="after", max_epochs=None):
    L.seed_everything(0, verbose=False)
    model = Tiny()
    es = EarlyStopping(monitor="val_loss", patience=4, mode="min") if with_es else None
    cbs = [es] if es else []
    cbs = cbs + list(extra) if order == "after" else list(extra) + cbs
    trainer = L.Trainer(
        max_epochs=max_epochs or MAX_EPOCHS, accelerator="cpu", devices=1, logger=False,
        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
        deterministic=True, callbacks=cbs, default_root_dir=str(root),
    )
    train_dl, val_dl = _loaders()
    trainer.fit(model, train_dl, val_dl)
    return {
        "current_epoch": trainer.current_epoch,
        "global_step": trainer.global_step,
        "stopped_epoch": es.stopped_epoch if es else None,
        "best_score": float(es.best_score) if es else None,
        "metrics": {k: float(v) for k, v in trainer.callback_metrics.items()},
        "state": {k: v.clone() for k, v in model.state_dict().items()},
    }


def _assert_identical(a, b):
    assert a["current_epoch"] == b["current_epoch"]
    assert a["global_step"] == b["global_step"]
    assert a["stopped_epoch"] == b["stopped_epoch"]
    assert a["best_score"] == b["best_score"]
    assert a["metrics"] == b["metrics"]
    assert a["state"].keys() == b["state"].keys()
    for k in a["state"]:
        assert torch.equal(a["state"][k], b["state"][k]), k


def test_control_seeded_runs_are_reproducible(tmp_path):
    a, b = _run(tmp_path), _run(tmp_path)
    assert 0 < a["stopped_epoch"] < MAX_EPOCHS - 1, "EarlyStopping must actually fire"
    _assert_identical(a, b)


@pytest.mark.parametrize("order", ["after", "before"])
def test_behaviour_unchanged_with_early_stopping(order, tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base = _run(tmp_path)
    capsys.readouterr()
    cb = SelfCareCallback(seed=0)
    comp = _run(tmp_path, [cb], order=order)
    out = capsys.readouterr().out
    assert base["stopped_epoch"] > 0
    _assert_identical(base, comp)
    assert out.count(PREFIX) == 1
    assert cb.last_roast and cb.last_roast in out
    # Inherited the EarlyStopping config and agrees on where it stopped (1-based).
    assert cb._companion.monitor == "val_loss" and cb._companion.patience == 4
    assert cb._companion.tracker.last_epoch == base["stopped_epoch"] + 1
    assert cb._companion.tracker.best == pytest.approx(base["best_score"])
    assert list(tmp_path.iterdir()) == []  # nothing written


def test_behaviour_unchanged_without_early_stopping(tmp_path, capsys):
    base = _run(tmp_path, with_es=False, max_epochs=25)
    capsys.readouterr()
    cb = SelfCareCallback(seed=0)
    comp = _run(tmp_path, [cb], with_es=False, max_epochs=25)
    _assert_identical(base, comp)
    assert comp["current_epoch"] == 25
    assert capsys.readouterr().out.count(PREFIX) == 1


def test_silent_mode_prints_nothing(tmp_path, capsys):
    cb = SelfCareCallback(seed=0, silent=True)
    res = _run(tmp_path, [cb])
    assert res["stopped_epoch"] > 0
    assert PREFIX not in capsys.readouterr().out
    assert cb.last_roast


def test_seeded_roast_is_deterministic(tmp_path):
    a, b = SelfCareCallback(seed=42, silent=True), SelfCareCallback(seed=42, silent=True)
    _run(tmp_path, [a])
    _run(tmp_path, [b])
    assert a.last_roast == b.last_roast


def test_callback_adds_nothing_to_checkpoints():
    assert SelfCareCallback().state_dict() == {}
