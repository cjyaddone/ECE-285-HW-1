"""Dataset and preprocessing utilities for the chest X-ray dataset."""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Callable

import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode


CLASS_NAMES = ("NORMAL", "PNEUMONIA")
IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}


class ResizeWithPadding:
    """Resize without changing anatomy proportions, then pad to a square."""

    def __init__(self, image_size: int) -> None:
        self.image_size = image_size

    def __call__(self, image: Image.Image) -> Image.Image:
        return ImageOps.pad(
            image,
            (self.image_size, self.image_size),
            method=Image.Resampling.BILINEAR,
            color=0,
            centering=(0.5, 0.5),
        )


class ChestXrayDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[Path, int]],
        transform: Callable[[Image.Image], torch.Tensor],
    ) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("L")
            image_tensor = self.transform(image)
        return image_tensor, label


def find_samples(split_dir: Path) -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []
    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = split_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        paths = sorted(
            path
            for path in class_dir.iterdir()
            if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        samples.extend((path, label) for path in paths)
    return samples


def stratified_split(
    samples: list[tuple[Path, int]], val_fraction: float, seed: int
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    rng = random.Random(seed)
    train_samples: list[tuple[Path, int]] = []
    val_samples: list[tuple[Path, int]] = []

    for label in range(len(CLASS_NAMES)):
        class_samples = [sample for sample in samples if sample[1] == label]
        rng.shuffle(class_samples)
        num_val = max(1, round(len(class_samples) * val_fraction))
        val_samples.extend(class_samples[:num_val])
        train_samples.extend(class_samples[num_val:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


def build_transform(image_size: int, augment: bool) -> transforms.Compose:
    operations: list[Callable] = [ResizeWithPadding(image_size)]
    if augment:
        operations.extend(
            [
                transforms.RandomAffine(
                    degrees=5,
                    translate=(0.03, 0.03),
                    scale=(0.95, 1.05),
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0,
                ),
                transforms.ColorJitter(brightness=0.08, contrast=0.08),
            ]
        )
    operations.append(transforms.ToTensor())
    return transforms.Compose(operations)


def make_loaders(
    data_dir: Path,
    image_size: int,
    batch_size: int,
    val_fraction: float,
    workers: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    all_train_samples = find_samples(data_dir / "train")
    train_samples, val_samples = stratified_split(all_train_samples, val_fraction, seed)

    train_dataset = ChestXrayDataset(
        train_samples, build_transform(image_size, augment=True)
    )
    val_dataset = ChestXrayDataset(
        val_samples, build_transform(image_size, augment=False)
    )

    class_counts = Counter(label for _, label in train_samples)
    sample_weights = [1.0 / class_counts[label] for _, label in train_samples]
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )

    common = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        sampler=sampler,
        drop_last=False,
        **common,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **common,
    )

    split_sizes = {
        "train": len(train_dataset),
        "validation": len(val_dataset),
        "train_normal": class_counts[0],
        "train_pneumonia": class_counts[1],
    }
    return train_loader, val_loader, split_sizes


def make_test_loader(
    data_dir: Path,
    image_size: int,
    batch_size: int,
    workers: int,
) -> DataLoader:
    dataset = ChestXrayDataset(
        find_samples(data_dir / "test"),
        build_transform(image_size, augment=False),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )
