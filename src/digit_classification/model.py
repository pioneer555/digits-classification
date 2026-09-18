import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torch import nn
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score


class DigitClassifier(LightningModule):
    """Tiny CNN for 0/5/8. Two conv stacks is plenty for this."""

    def __init__(
        self,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        class_weights: list[float] | None = None,
        num_classes: int = 3,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.save_hyperparameters()

        # 28 -> 14 -> 7
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        if class_weights is None:
            self.class_weights = None
        else:
            self.register_buffer(
                "class_weights", torch.tensor(class_weights, dtype=torch.float32)
            )

        self.train_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.val_acc = MulticlassAccuracy(num_classes=num_classes, average="micro")

    def forward(self, x):
        return self.classifier(self.features(x))

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

    def _shared_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y, weight=self.class_weights)
        return loss, logits.argmax(1), y

    def training_step(self, batch, batch_idx=0):
        loss, preds, y = self._shared_step(batch)
        self.train_f1(preds, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_f1", self.train_f1, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx=0):
        loss, preds, y = self._shared_step(batch)
        self.val_f1(preds, y)
        self.val_acc(preds, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx=0):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        return torch.softmax(self(x), dim=1)
