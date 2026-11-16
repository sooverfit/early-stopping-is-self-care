"""Same equivalence proof for the legacy ``pytorch_lightning`` package name."""

import pytest

torch = pytest.importorskip("torch")
pl = pytest.importorskip("pytorch_lightning")

from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from early_stopping_is_self_care.core import PREFIX  # noqa: E402
from early_stopping_is_self_care.pytorch_lightning import SelfCareCallback  # noqa: E402


class Tiny(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(8, 64), torch.nn.ReLU(),
                                       torch.nn.Linear(64, 1))

    def training_step(self, batch, _):
        x, y = batch
        return torch.nn.functional.mse_loss(self.net(x), y)

    def validation_step(self, batch, _):
        x, y = batch
        self.log("val_loss", torch.nn.functional.mse_loss(self.net(x), y))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)


def _run(root, extra=()):
    pl.seed_everything(0, verbose=False)
    g = torch.Generator().manual_seed(0)
    w = torch.randn(8, 1, generator=g)
    x, xv = torch.randn(48, 8, generator=g), torch.randn(64, 8, generator=g)
    y, yv = x @ w + 1.5 * torch.randn(48, 1, generator=g), xv @ w + 1.5 * torch.randn(64, 1, generator=g)
    es = pl.callbacks.EarlyStopping(monitor="val_loss", patience=4)
    model = Tiny()
    trainer = pl.Trainer(max_epochs=200, accelerator="cpu", logger=False,
                         enable_checkpointing=False, enable_progress_bar=False,
                         enable_model_summary=False, deterministic=True,
                         callbacks=[es, *extra], default_root_dir=str(root))
    trainer.fit(model, DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True),
                DataLoader(TensorDataset(xv, yv), batch_size=64))
    return es.stopped_epoch, trainer.current_epoch, model.state_dict()


def test_behaviour_unchanged_pytorch_lightning(tmp_path, capsys):
    a = _run(tmp_path)
    capsys.readouterr()
    cb = SelfCareCallback(seed=0)
    b = _run(tmp_path, [cb])
    assert a[0] > 0 and a[:2] == b[:2]
    assert all(torch.equal(a[2][k], b[2][k]) for k in a[2])
    assert capsys.readouterr().out.count(PREFIX) == 1 and cb.last_roast
