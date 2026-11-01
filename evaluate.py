"""Compute FID and Inception Score for generated X-rays with PyTorch-Ignite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from ignite.metrics import FID, InceptionScore

from data import make_test_loader
from model import ConditionalVAE


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=project_dir / "outputs" / "spatial_film" / "best.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_dir / "outputs" / "spatial_film" / "metrics.json",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=624,
        help="Maximum number of held-out real/generated pairs to evaluate.",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
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


def prepare_for_inception(images: torch.Tensor) -> torch.Tensor:
    """Convert grayscale 128 x 128 images to Inception's RGB input shape."""
    images = images.repeat(1, 3, 1, 1)
    return F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)


def main() -> None:
    args = parse_args()
    if args.num_samples < 2:
        raise ValueError("num_samples must be at least 2 for FID")

    torch.manual_seed(args.seed)
    device = choose_device(args.device)
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

    test_loader = make_test_loader(
        Path(config["data_dir"]),
        image_size=config["image_size"],
        batch_size=args.batch_size,
        workers=args.workers,
    )
    num_available = len(test_loader.dataset)
    target_count = min(args.num_samples, num_available)

    # Ignite's default feature extractor is ImageNet Inception v3. Inputs must
    # therefore be RGB and 299 x 299 even though the CVAE itself is grayscale.
    fid = FID(device=device)
    inception_score = InceptionScore(device=device)
    fid.reset()
    inception_score.reset()

    num_processed = 0
    with torch.no_grad():
        for real_images, labels in test_loader:
            remaining = target_count - num_processed
            if remaining <= 0:
                break
            real_images = real_images[:remaining].to(device)
            labels = labels[:remaining].to(device)
            generated_images = model.sample(labels)

            real_rgb = prepare_for_inception(real_images)
            generated_rgb = prepare_for_inception(generated_images)
            fid.update((generated_rgb, real_rgb))
            inception_score.update(generated_rgb)
            num_processed += real_images.shape[0]
            print(f"Evaluated {num_processed}/{target_count} images", end="\r")

    results = {
        "checkpoint": str(args.checkpoint.resolve()),
        "num_samples": num_processed,
        "fid": fid.compute(),
        "inception_score": inception_score.compute(),
        "note": (
            "ImageNet Inception features are intended for relative comparison "
            "between experiments, not clinical validation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nFID: {results['fid']:.4f}")
    print(f"Inception Score: {results['inception_score']:.4f}")
    print(f"Saved metrics to {args.output.resolve()}")


if __name__ == "__main__":
    main()
