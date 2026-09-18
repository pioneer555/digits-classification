import torch
from torch.utils.data import DataLoader, TensorDataset

from digit_classification.evaluation import (
    collect_predictions,
    evaluate_model,
    format_confusion_matrix,
    generate_classification_report,
    generate_confusion_matrix,
    mapped_to_digits,
)
from digit_classification.model import DigitClassifier


def test_mapped_to_digits():
    assert mapped_to_digits([0, 1, 2, 1]) == [0, 5, 8, 5]
    assert mapped_to_digits(torch.tensor([2, 0])) == [8, 0]


def test_classification_report_includes_all_digits():
    report = generate_classification_report(
        [0, 5, 8, 0, 5, 8],
        [0, 5, 5, 0, 5, 8],
    )
    for token in ("0", "5", "8", "precision", "recall"):
        assert token in report


def test_confusion_matrix_shape_and_values():
    matrix = generate_confusion_matrix([0, 0, 5, 8], [0, 5, 5, 8])
    assert matrix == [
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
    text = format_confusion_matrix(matrix)
    assert "true\\pred" in text
    assert "5" in text


def test_evaluate_model_on_dummy_network():
    model = DigitClassifier()
    model.eval()
    x = torch.randn(9, 1, 28, 28)
    y = torch.arange(9) % 3
    loader = DataLoader(TensorDataset(x, y), batch_size=3)
    report, matrix = evaluate_model(model, loader)
    assert "precision" in report
    assert "true\\pred" in matrix
    yt, yp = collect_predictions(model, loader)
    assert len(yt) == len(yp) == 9
