from pathlib import Path

import pytest
import torch
from PIL import Image
from typer.testing import CliRunner

from digit_classification.cli import app
from digit_classification.data import (
    DIGIT_LABELS,
    LABEL_TO_INDEX,
    compute_class_weights,
    create_dataloader,
    curate_subset,
    load_image_tensor,
    prepare_splits,
    stratified_split,
    summarize_labels,
)


def _fake_mnist(n=50):
    imgs, ys = [], []
    for d in range(10):
        imgs.append(torch.full((n, 28, 28), d, dtype=torch.uint8))
        ys.append(torch.full((n,), d, dtype=torch.long))
    return torch.cat(imgs), torch.cat(ys)


def test_curate_subset_counts_and_labels():
    images, targets = _fake_mnist(40)
    counts = {0: 20, 5: 10, 8: 30}
    subset, original, mapped = curate_subset(images, targets, counts=counts, seed=0)

    assert subset.shape == (60, 28, 28)
    assert summarize_labels(original) == counts
    assert set(original.tolist()) == set(DIGIT_LABELS)
    assert set(mapped.tolist()) == {0, 1, 2}
    for y, m in zip(original.tolist(), mapped.tolist()):
        assert LABEL_TO_INDEX[y] == m


def test_curate_subset_is_reproducible():
    images, targets = _fake_mnist(40)
    counts = {0: 12, 5: 8, 8: 16}
    a = curate_subset(images, targets, counts=counts, seed=123)
    b = curate_subset(images, targets, counts=counts, seed=123)
    c = curate_subset(images, targets, counts=counts, seed=124)

    assert torch.equal(a[0], b[0])
    assert torch.equal(a[1], b[1])
    assert not torch.equal(a[0], c[0])


def test_curate_subset_rejects_insufficient_samples():
    images, targets = _fake_mnist(5)
    with pytest.raises(ValueError, match="digit 0"):
        curate_subset(images, targets, counts={0: 20, 5: 1, 8: 1}, seed=0)


def test_stratified_split_is_reproducible_and_balanced():
    images, targets = _fake_mnist(50)
    subset, original, mapped = curate_subset(
        images, targets, counts={0: 20, 5: 20, 8: 20}, seed=1
    )
    train1, test1 = stratified_split(subset, mapped, original, 0.25, seed=7)
    train2, test2 = stratified_split(subset, mapped, original, 0.25, seed=7)

    assert torch.equal(test1[1], test2[1])
    assert len(test1[1]) == 15
    assert len(train1[1]) == 45
    assert set(test1[1].tolist()) == {0, 1, 2}
    assert set(train1[1].tolist()) == {0, 1, 2}


def test_class_weights_upweight_rare_class():
    labels = torch.tensor([0] * 8 + [1] * 2 + [2] * 10)
    w = compute_class_weights(labels)
    assert w[1] > w[0]
    assert w[1] > w[2]


def test_dataloader_shapes():
    images = torch.randint(0, 255, (10, 28, 28), dtype=torch.uint8)
    labels = torch.arange(10) % 3
    x, y = next(iter(create_dataloader(images, labels, batch_size=4, shuffle=False)))
    assert x.shape == (4, 1, 28, 28)
    assert y.shape == (4,)


def test_prepare_splits_uses_holdout_and_is_reproducible(monkeypatch):
    images, targets = _fake_mnist(80)
    fake = type("FakeMNIST", (), {"data": images, "targets": targets})()
    monkeypatch.setattr("digit_classification.data.load_mnist", lambda *a, **k: fake)

    counts = {0: 40, 5: 20, 8: 40}
    a = prepare_splits("unused", seed=3, counts=counts)
    b = prepare_splits("unused", seed=3, counts=counts)

    total = sum(len(a[s][1]) for s in ("train", "val", "test"))
    assert total == 100
    assert len(a["test"][1]) == 20
    assert torch.equal(a["test"][1], b["test"][1])
    assert set(a["train"][1].tolist()) == {0, 1, 2}


def test_cli_help_and_epoch_limit():
    runner = CliRunner()
    help_out = runner.invoke(app, ["--help"])
    assert help_out.exit_code == 0
    assert "download-data" in help_out.stdout
    assert "train" in help_out.stdout

    too_many = runner.invoke(
        app, ["train", "--data-dir", "data", "--output-dir", "out", "--epochs", "21"]
    )
    assert too_many.exit_code != 0


def test_load_image_inverts_dark_ink_on_white(tmp_path: Path):
    path = tmp_path / "digit.png"
    Image.new("L", (28, 28), color=255).save(path)
    t = load_image_tensor(path)
    assert t.shape == (1, 1, 28, 28)
    # white page -> invert -> mostly zeros -> negative after MNIST norm
    assert t.mean().item() < 0
