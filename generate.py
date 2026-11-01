"""Generate class-conditional samples from a trained CVAE checkpoint."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torchvision.utils import save_image

from data import CLASS_NAMES, make_test_loader
from model import ConditionalVAE


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=project_dir / "outputs" / "spatial_film" / "best.pt",
    )
    parser.add_argument("--output-dir", type=Path, default=project_dir / "generated")
    parser.add_argument("--num-per-class", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--device", choices=("auto", "cuda", "mps", "cpu"), default="auto"
    )
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = ConditionalVAE(
        image_size=config["image_size"],
        latent_dim=config["latent_dim"],
        num_classes=config["num_classes"],
        condition_dim=config["condition_dim"],
        architecture=config.get("architecture", "bottleneck"),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    all_samples = []
    for label, class_name in enumerate(CLASS_NAMES):
        labels = torch.full(
            (args.num_per_class,), label, dtype=torch.long, device=device
        )
        with torch.no_grad():
            samples = model.sample(labels).cpu()
        all_samples.append(samples)

        class_dir = args.output_dir / class_name.lower()
        class_dir.mkdir(exist_ok=True)
        for index, sample in enumerate(samples):
            save_image(sample, class_dir / f"sample_{index:03d}.png")

    sample_grid = torch.cat(all_samples)
    save_image(
        sample_grid,
        args.output_dir / "conditional_samples.png",
        nrow=args.num_per_class,
        padding=2,
    )

    test_loader = make_test_loader(
        Path(config["data_dir"]),
        image_size=config["image_size"],
        batch_size=8,
        workers=args.workers,
    )
    examples_by_label: dict[int, list[torch.Tensor]] = {0: [], 1: []}
    for batch_images, batch_labels in test_loader:
        for image, label in zip(batch_images, batch_labels):
            label_value = int(label)
            if len(examples_by_label[label_value]) < 4:
                examples_by_label[label_value].append(image)
        if all(len(examples) == 4 for examples in examples_by_label.values()):
            break
    images = torch.stack(examples_by_label[0] + examples_by_label[1])
    labels = torch.tensor([0] * 4 + [1] * 4)
    with torch.no_grad():
        mu, _ = model.encode(images.to(device), labels.to(device))
        reconstructions = model.decode(mu, labels.to(device))
    comparison = torch.cat((images, reconstructions.cpu()))
    save_image(
        comparison,
        args.output_dir / "test_reconstructions.png",
        nrow=8,
        padding=2,
    )
    print(f"Saved generated images to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
