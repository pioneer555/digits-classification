from pathlib import Path

import torch
import typer
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint

from digit_classification.data import (
    DEFAULT_SEED,
    DIGIT_LABELS,
    INDEX_TO_LABEL,
    compute_class_weights,
    create_dataloader,
    download_mnist,
    load_image_tensor,
    prepare_splits,
    summarize_labels,
)
from digit_classification.evaluation import evaluate_model
from digit_classification.model import DigitClassifier

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command("download-data")
def download_data(data_dir: str = typer.Option(..., "--data-dir")):
    """Grab the MNIST training set."""
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    ds = download_mnist(data_dir)
    typer.echo(f"got {len(ds)} images in {data_dir}")


@app.command()
def train(
    data_dir: str = typer.Option(..., "--data-dir"),
    output_dir: str = typer.Option(..., "--output-dir"),
    epochs: int = typer.Option(20, "--epochs"),
    batch_size: int = typer.Option(64, "--batch-size"),
    lr: float = typer.Option(1e-3, "--lr"),
    seed: int = typer.Option(DEFAULT_SEED, "--seed"),
):
    """Fit the 0/5/8 CNN. Caps at 20 epochs."""
    if epochs > 20:
        raise typer.BadParameter("max 20 epochs")

    seed_everything(seed, workers=False)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    splits = prepare_splits(data_dir, seed=seed)
    train_x, train_y, train_digits = splits["train"]
    val_x, val_y, val_digits = splits["val"]

    typer.echo(f"train {summarize_labels(train_digits)}")
    typer.echo(f"val   {summarize_labels(val_digits)}")
    typer.echo(f"test  {summarize_labels(splits['test'][2])}")

    weights = compute_class_weights(train_y)
    typer.echo(f"weights [0, 5, 8]: {[round(w, 4) for w in weights]}")

    model = DigitClassifier(lr=lr, class_weights=weights)
    ckpt = ModelCheckpoint(
        dirpath=output_dir,
        filename="digit-classifier-{epoch:02d}-{val_f1:.3f}",
        monitor="val_f1",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    trainer = Trainer(
        default_root_dir=output_dir,
        accelerator="cpu",
        max_epochs=epochs,
        callbacks=[ckpt],
        log_every_n_steps=10,
        enable_model_summary=True,
    )
    trainer.fit(
        model,
        train_dataloaders=create_dataloader(train_x, train_y, batch_size, shuffle=True, seed=seed),
        val_dataloaders=create_dataloader(val_x, val_y, batch_size, shuffle=False, seed=seed),
    )

    best = ckpt.best_model_path or str(Path(output_dir) / "last.ckpt")
    typer.echo(f"best: {best}")
    typer.echo(f"last: {Path(output_dir) / 'last.ckpt'}")


@app.command()
def evaluate(
    checkpoint_path: str = typer.Option(..., "--checkpoint-path"),
    data_dir: str = typer.Option(..., "--data-dir"),
    batch_size: int = typer.Option(64, "--batch-size"),
    seed: int = typer.Option(DEFAULT_SEED, "--seed"),
):
    """sklearn report + confusion matrix on the hold-out set."""
    seed_everything(seed, workers=False)
    splits = prepare_splits(data_dir, seed=seed)
    test_x, test_y, test_digits = splits["test"]
    typer.echo(f"test {summarize_labels(test_digits)}")

    model = DigitClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()
    report, matrix = evaluate_model(
        model,
        create_dataloader(test_x, test_y, batch_size, shuffle=False, seed=seed),
    )
    typer.echo("\nClassification report")
    typer.echo(report)
    typer.echo("Confusion matrix")
    typer.echo(matrix)


@app.command()
def predict(
    checkpoint_path: str = typer.Option(..., "--checkpoint-path"),
    input_path: str = typer.Option(..., "--input-path"),
):
    """Class probs for one image (0 / 5 / 8)."""
    model = DigitClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()
    x = load_image_tensor(input_path)
    with torch.no_grad():
        probs = model.predict_step(x).squeeze(0)

    pred = INDEX_TO_LABEL[int(probs.argmax().item())]
    typer.echo(f"Predicted digit: {pred}")
    for digit, p in zip(DIGIT_LABELS, probs.tolist()):
        typer.echo(f"  P({digit}) = {p:.4f}")


if __name__ == "__main__":
    app()
