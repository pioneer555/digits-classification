from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST

# class ids are 0/1/2 internally; these are the actual digits
DIGIT_LABELS = (0, 5, 8)
LABEL_TO_INDEX = {0: 0, 5: 1, 8: 2}
INDEX_TO_LABEL = {v: k for k, v in LABEL_TO_INDEX.items()}

# 5k images, lots of 8s, almost no 5s
TARGET_COUNTS = {0: 1200, 5: 300, 8: 3500}

DEFAULT_SEED = 42
TEST_FRACTION = 0.2
VAL_FRACTION = 0.2  # of the leftover after pulling test
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


def download_mnist(data_dir: str | Path) -> MNIST:
    return MNIST(root=str(data_dir), train=True, download=True)


def load_mnist(data_dir: str | Path, download: bool = False) -> MNIST:
    return MNIST(root=str(data_dir), train=True, download=download)


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ]
    )


def _select_indices(targets, label, count, generator):
    hits = (targets == label).nonzero(as_tuple=True)[0]
    if hits.numel() < count:
        raise ValueError(f"need {count} of digit {label}, only have {hits.numel()}")
    order = torch.randperm(hits.numel(), generator=generator)
    return hits[order[:count]]


def curate_subset(images, targets, counts=None, seed=DEFAULT_SEED):
    counts = counts or TARGET_COUNTS
    g = torch.Generator().manual_seed(seed)

    picked = [_select_indices(targets, label, counts[label], g) for label in DIGIT_LABELS]
    idx = torch.cat(picked)
    idx = idx[torch.randperm(idx.numel(), generator=g)]  # don't leave them grouped by class

    original = targets[idx]
    mapped = torch.tensor([LABEL_TO_INDEX[int(y)] for y in original], dtype=torch.long)
    return images[idx], original, mapped


def stratified_split(images, mapped_labels, original_labels, test_size, seed):
    idx = list(range(len(mapped_labels)))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        stratify=mapped_labels.numpy(),
        random_state=seed,
    )
    train_idx = torch.tensor(train_idx, dtype=torch.long)
    test_idx = torch.tensor(test_idx, dtype=torch.long)
    train = (images[train_idx], mapped_labels[train_idx], original_labels[train_idx])
    test = (images[test_idx], mapped_labels[test_idx], original_labels[test_idx])
    return train, test


def prepare_splits(data_dir, seed=DEFAULT_SEED, counts=None, download=False):
    mnist = load_mnist(data_dir, download=download)
    images, original, mapped = curate_subset(mnist.data, mnist.targets, counts=counts, seed=seed)

    train_val, test = stratified_split(images, mapped, original, TEST_FRACTION, seed)
    train, val = stratified_split(train_val[0], train_val[1], train_val[2], VAL_FRACTION, seed)
    return {"train": train, "val": val, "test": test}


def compute_class_weights(mapped_labels, num_classes=3):
    counts = torch.bincount(mapped_labels, minlength=num_classes).float()
    # n / (c * n_i), then it averages to 1
    weights = counts.sum() / (num_classes * counts.clamp(min=1.0))
    return weights.tolist()


class DigitDataset(Dataset):
    def __init__(self, images, mapped_labels, transform=None):
        self.images = images
        self.mapped_labels = mapped_labels
        self.transform = transform or build_transform()

    def __len__(self):
        return len(self.mapped_labels)

    def __getitem__(self, i):
        img = Image.fromarray(self.images[i].numpy(), mode="L")
        return self.transform(img), self.mapped_labels[i]


def create_dataloader(images, mapped_labels, batch_size, shuffle, seed=DEFAULT_SEED):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(
        DigitDataset(images, mapped_labels),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=g,
    )


def load_image_tensor(image_path: str | Path) -> torch.Tensor:
    image = Image.open(image_path).convert("L")
    to_tensor = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
        ]
    )
    raw = to_tensor(image)
    # MNIST is white-on-black. A photo of ink on paper is the opposite.
    if raw.mean() > 0.5:
        raw = 1.0 - raw
    return transforms.Normalize(MNIST_MEAN, MNIST_STD)(raw).unsqueeze(0)


def summarize_labels(original_labels):
    return dict(sorted(Counter(original_labels.tolist()).items()))
