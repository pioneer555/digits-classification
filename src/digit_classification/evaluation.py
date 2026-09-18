import torch
from sklearn.metrics import classification_report, confusion_matrix

from digit_classification.data import DIGIT_LABELS, INDEX_TO_LABEL


def mapped_to_digits(mapped_labels):
    vals = mapped_labels.tolist() if isinstance(mapped_labels, torch.Tensor) else mapped_labels
    return [INDEX_TO_LABEL[int(y)] for y in vals]


def collect_predictions(model, dataloader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in dataloader:
            y_true.extend(y.tolist())
            y_pred.extend(model(x).argmax(1).tolist())
    return y_true, y_pred


def generate_classification_report(y_true_digits, y_pred_digits):
    return classification_report(
        y_true_digits,
        y_pred_digits,
        labels=list(DIGIT_LABELS),
        digits=4,
        zero_division=0,
    )


def generate_confusion_matrix(y_true_digits, y_pred_digits):
    return confusion_matrix(
        y_true_digits,
        y_pred_digits,
        labels=list(DIGIT_LABELS),
    ).tolist()


def format_confusion_matrix(matrix):
    header = "true\\pred  " + "  ".join(f"{d:>4}" for d in DIGIT_LABELS)
    lines = [header]
    for digit, row in zip(DIGIT_LABELS, matrix):
        cells = "  ".join(f"{n:>4}" for n in row)
        lines.append(f"{digit:>9}  {cells}")
    return "\n".join(lines)


def evaluate_model(model, dataloader):
    y_true, y_pred = collect_predictions(model, dataloader)
    y_true = mapped_to_digits(y_true)
    y_pred = mapped_to_digits(y_pred)
    report = generate_classification_report(y_true, y_pred)
    matrix = format_confusion_matrix(generate_confusion_matrix(y_true, y_pred))
    return report, matrix
