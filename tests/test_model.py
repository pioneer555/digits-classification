import torch
from lightning.pytorch import Trainer, seed_everything
from torch.utils.data import DataLoader, TensorDataset

from digit_classification.model import DigitClassifier


def test_forward_output_shape():
    logits = DigitClassifier()(torch.randn(4, 1, 28, 28))
    assert logits.shape == (4, 3)


def test_predict_step_returns_probabilities():
    model = DigitClassifier()
    model.eval()
    x = torch.randn(5, 1, 28, 28)
    probs = model.predict_step((x, torch.zeros(5, dtype=torch.long)))
    assert probs.shape == (5, 3)
    assert torch.allclose(probs.sum(1), torch.ones(5), atol=1e-5)
    assert torch.all(probs >= 0)


def test_predict_step_accepts_images_only():
    probs = DigitClassifier().predict_step(torch.randn(2, 1, 28, 28))
    assert probs.shape == (2, 3)


def test_shared_step_loss_is_finite():
    model = DigitClassifier(class_weights=[0.5, 2.0, 0.7])
    loss, preds, y = model._shared_step(
        (torch.randn(6, 1, 28, 28), torch.tensor([0, 1, 2, 0, 1, 2]))
    )
    assert torch.isfinite(loss)
    assert preds.shape == y.shape


def test_configure_optimizers():
    opt = DigitClassifier(lr=3e-4).configure_optimizers()
    assert opt.param_groups[0]["lr"] == 3e-4


def test_one_epoch_cpu_training_runs():
    seed_everything(0, workers=False)
    model = DigitClassifier(class_weights=[1.0, 1.5, 0.8])
    x = torch.randn(16, 1, 28, 28)
    y = torch.arange(16) % 3
    loader = DataLoader(TensorDataset(x, y), batch_size=8)
    trainer = Trainer(
        accelerator="cpu",
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    trainer.fit(model, loader, loader)
    assert trainer.callback_metrics["train_loss"].isfinite()
