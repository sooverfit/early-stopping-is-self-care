import builtins
import os
import pathlib
import random
import re
import socket
import string

import pytest

from early_stopping_is_self_care import Companion, PlateauTracker, compose_roast, diagnose, roast
from early_stopping_is_self_care import roasts as R
from early_stopping_is_self_care.core import PREFIX, resolve_mode

from conftest import NetworkBlocked

PKG = pathlib.Path(__file__).resolve().parents[1] / "src" / "early_stopping_is_self_care"
ALL = R.GENERAL + R.STOPPED + R.PASSED_PEAK
FIELDS = {"best_epoch", "last_epoch", "extra_epochs", "monitor", "best", "patience"}


def test_there_are_about_thirty_unique_jokes():
    assert len(ALL) >= 30
    assert len(set(ALL)) == len(ALL)


@pytest.mark.parametrize("template", ALL)
def test_every_joke_formats_and_is_short(template):
    used = {f for _, f, _, _ in string.Formatter().parse(template) if f}
    assert used <= FIELDS
    text = template.format(best_epoch=22, last_epoch=27, extra_epochs="5 epochs",
                           monitor="val_loss", best="0.1234", patience=5)
    assert "{" not in text and len(text) <= 200


def test_general_and_passed_peak_never_claim_early_stopping_fired():
    for t in R.GENERAL + R.PASSED_PEAK:
        assert not re.search(r"early stopping (fired|noticed|respected|called)", t, re.I)


def test_seed_is_deterministic_and_varies():
    a = [roast(seed=s) for s in range(20)]
    b = [roast(seed=s) for s in range(20)]
    assert a == b
    assert len(set(a)) > 5


def test_compose_roast_uses_numbers_and_pluralises():
    texts = {compose_roast(stopped=True, best_epoch=3, last_epoch=4, rng=random.Random(s))
             for s in range(200)}
    assert not any("1 epochs" in t for t in texts)
    assert compose_roast(stopped=True, best_epoch=4, last_epoch=4) == R.FALLBACK


def test_tracker_min_delta_semantics():
    t = PlateauTracker("val_loss", "min", min_delta=0.1)
    for e, v in enumerate([1.0, 0.95, 0.85, 0.9], start=1):
        t.update(e, v)
    # 0.95 is not < 1.0 - 0.1, 0.85 is.
    assert (t.best_epoch, t.best, t.epochs_since_best) == (3, 0.85, 1)
    assert resolve_mode("val_accuracy") == "max"
    assert resolve_mode("val_loss") == "min"


def test_companion_prints_once_and_silent(capsys):
    c = Companion(seed=1)
    c.begin()
    for e, v in enumerate([3, 2, 1, 2, 3], start=1):
        c.observe(e, v)
    first = c.finish(stopped=True)
    assert c.finish(stopped=True) == first
    assert capsys.readouterr().out.count(PREFIX) == 1
    s = Companion(seed=1, silent=True)
    s.begin()
    for e, v in enumerate([3, 2, 1, 2, 3], start=1):
        s.observe(e, v)
    assert s.finish(stopped=True) == first
    assert capsys.readouterr().out == ""


def test_env_var_silences(capsys, monkeypatch):
    monkeypatch.setenv("ESISC_SILENT", "1")
    c = Companion(seed=1)
    c.begin()
    c.observe(1, 1.0)
    c.observe(2, 2.0)
    assert c.finish(stopped=True)
    assert capsys.readouterr().out == ""


def test_no_roast_when_run_ends_on_its_best_epoch(capsys):
    c = Companion()
    c.begin()
    for e, v in enumerate([3, 2, 1], start=1):
        c.observe(e, v)
    assert c.finish(stopped=False) is None
    assert capsys.readouterr().out == ""


def _curve(kind, n=30):
    rng = random.Random(0)
    train, val = [], []
    for i in range(n):
        t = i / (n - 1)
        if kind == "overfitting":
            train.append(1.1 * 2.718 ** (-3.2 * t) + 0.08)
            val.append(1.0 * 2.718 ** (-2.4 * t) + 0.18 + 0.85 * max(0, t - 0.35) ** 2)
        elif kind == "underfitting":
            train.append(1.6 - 0.7 * t)
            val.append(1.7 - 0.65 * t)
        elif kind == "healthy":
            p = min(t / 0.55, 1)
            train.append(1.2 * 2.718 ** (-3.2 * p) + 0.22)
            val.append(1.15 * 2.718 ** (-2.8 * p) + 0.28)
        else:
            train.append(1.1 * 2.718 ** (-2.5 * t) + 0.2)
            val.append(1.0 * 2.718 ** (-2.2 * t) + 0.28 + rng.gauss(0, 0.11))
    return train, val


@pytest.mark.parametrize("kind", ["overfitting", "underfitting", "healthy", "noisy"])
def test_diagnose_examples(kind):
    d = diagnose(*_curve(kind), patience=5, seed=0)
    assert d.verdict == kind
    assert str(d).startswith(f"Verdict: {kind}")


def test_diagnose_overfit_details():
    d = diagnose(*_curve("overfitting"), patience=5, seed=0)
    assert d.early_stop_epoch == d.best_epoch + 5
    assert d.roast and str(d.best_epoch) in str(d)
    with pytest.raises(ValueError):
        diagnose([1, 2], [1])


def test_network_guard_is_active():
    with pytest.raises(NetworkBlocked):
        socket.create_connection(("example.com", 80))
    with pytest.raises(NetworkBlocked):
        socket.socket().connect(("93.184.216.34", 80))


def test_source_has_no_network_or_file_write_code():
    banned = re.compile(r"^\s*(import|from)\s+(socket|ssl|http|urllib|requests|httpx|aiohttp|"
                        r"ftplib|smtplib|telnetlib|xmlrpc)\b", re.M)
    for path in PKG.glob("*.py"):
        src = path.read_text()
        assert not banned.search(src), path
        assert "open(" not in src, path
        assert "write_text" not in src and "write_bytes" not in src, path


def test_core_paths_touch_no_files(monkeypatch):
    def nope(*a, **k):
        raise AssertionError("file access attempted")

    monkeypatch.setattr(builtins, "open", nope)
    monkeypatch.setattr(os, "open", nope)
    diagnose(*_curve("overfitting"), seed=3)
    c = Companion(seed=3, printer=lambda s: None)
    c.begin()
    c.observe(1, 1.0)
    c.observe(2, 2.0)
    assert c.finish(stopped=True)
