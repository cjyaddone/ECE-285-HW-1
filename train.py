"""Train the conditional VAE and record losses in TensorBoard."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from data import CLASS_NAMES, make_loaders
from losses import cvae_loss
from model import ConditionalVAE


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_dir / "dataset" / "chest_xray" / "chest_xray",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=project_dir / "outputs" / "spatial_film"
    )
    parser.add_argument(
        "--log-dir", type=Path, default=project_dir / "runs" / "spatial_film"
    )
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--condition-dim", type=int, default=32)
    parser.add_argument(
        "--architecture",
        choices=("spatial_film", "bottleneck"),
        default="spatial_film",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--beta", type=float, default=3e-4)
    parser.add_argument("--kl-warmup-epochs", type=int, default=30)
    parser.add_argument("--ssim-weight", type=float, default=0.2)
    parser.add_argument("--edge-weight", type=float, default=0.1)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-log-interval", type=int, default=5)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def beta_for_epoch(epoch: int, beta_max: float, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return beta_max
    return beta_max * min(1.0, epoch / warmup_epochs)


def run_epoch(
    model: ConditionalVAE,
    loader: DataLoader,
    device: torch.device,
    beta: float,
    ssim_weight: float,
    edge_weight: float,
    optimizer: Adam | None,
    scaler: torch.amp.GradScaler | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {name: 0.0 for name in ("total", "l1", "ssim", "edge", "kl")}
    num_examples = 0
    use_amp = scaler is not None

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = images.shape[0]

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                if training:
                    reconstruction, mu, logvar = model(images, labels)
                else:
                    # Using the posterior mean makes validation reproducible.
                    mu, logvar = model.encode(images, labels)
                    reconstruction = model.decode(mu, labels)
                losses = cvae_loss(
                    reconstruction,
                    images,
                    mu,
                    logvar,
                    beta=beta,
                    ssim_weight=ssim_weight,
                    edge_weight=edge_weight,
                )

            if training:
                if scaler is None:
                    losses["total"].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
                else:
                    scaler.scale(losses["total"]).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()

        for name, value in losses.items():
            totals[name] += value.detach().item() * batch_size
        num_examples += batch_size

    return {name: value / num_examples for name, value in totals.items()}


@torch.no_grad()
def log_images(
    writer: SummaryWriter,
    model: ConditionalVAE,
    val_loader: DataLoader,
    device: torch.device,
    epoch: int,
) -> None:
    model.eval()
    images, labels = next(iter(val_loader))
    images = images[:8].to(device)
    labels = labels[:8].to(device)
    mu, _ = model.encode(images, labels)
    reconstructions = model.decode(mu, labels)

    comparison = torch.cat((images, reconstructions), dim=0)
    writer.add_image(
        "images/originals_then_reconstructions",
        make_grid(comparison.cpu(), nrow=8, padding=2),
        epoch,
    )

    sample_labels = torch.tensor([0, 1] * 4, device=device)
    samples = model.sample(sample_labels)
    writer.add_image(
        "images/conditional_samples_alternating_normal_pneumonia",
        make_grid(samples.cpu(), nrow=8, padding=2),
        epoch,
    )


def save_checkpoint(
    path: Path,
    model: ConditionalVAE,
    optimizer: Adam,
    scheduler: ReduceLROnPlateau,
    epoch: int,
    best_val_loss: float,
    args: argparse.Namespace,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "config": {
            "image_size": args.image_size,
            "latent_dim": args.latent_dim,
            "condition_dim": args.condition_dim,
            "architecture": args.architecture,
            "num_classes": len(CLASS_NAMES),
            "class_names": CLASS_NAMES,
            "data_dir": str(args.data_dir.resolve()),
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "beta": args.beta,
            "kl_warmup_epochs": args.kl_warmup_epochs,
            "ssim_weight": args.ssim_weight,
            "edge_weight": args.edge_weight,
            "val_fraction": args.val_fraction,
            "seed": args.seed,
        },
    }
    torch.save(checkpoint, path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    resume_checkpoint = None
    if args.resume is not None:
        resume_checkpoint = torch.load(
            args.resume, map_location=device, weights_only=False
        )
        resume_config = resume_checkpoint["config"]
        args.image_size = resume_config["image_size"]
        args.latent_dim = resume_config["latent_dim"]
        args.condition_dim = resume_config["condition_dim"]
        args.architecture = resume_config.get("architecture", "bottleneck")

    train_loader, val_loader, split_sizes = make_loaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        workers=args.workers,
        seed=args.seed,
    )
    print(f"Device: {device}")
    print(f"Dataset split: {split_sizes}")

    model = ConditionalVAE(
        image_size=args.image_size,
        latent_dim=args.latent_dim,
        num_classes=len(CLASS_NAMES),
        condition_dim=args.condition_dim,
        architecture=args.architecture,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    start_epoch = 1
    best_val_loss = float("inf")
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        if "scheduler_state" in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint["scheduler_state"])
        start_epoch = resume_checkpoint["epoch"] + 1
        best_val_loss = resume_checkpoint["best_val_loss"]
        print(f"Resumed from epoch {resume_checkpoint['epoch']}")

    writer = SummaryWriter(log_dir=args.log_dir)
    writer.add_text(
        "configuration", "\n".join(f"{k}: {v}" for k, v in vars(args).items())
    )

    for epoch in range(start_epoch, args.epochs + 1):
        beta = beta_for_epoch(epoch, args.beta, args.kl_warmup_epochs)
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            beta=beta,
            ssim_weight=args.ssim_weight,
            edge_weight=args.edge_weight,
            optimizer=optimizer,
            scaler=scaler,
        )
        # Always use the full beta for validation so early warm-up epochs do not
        # receive an artificially favorable validation objective.
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            beta=args.beta,
            ssim_weight=args.ssim_weight,
            edge_weight=args.edge_weight,
            optimizer=None,
            scaler=None,
        )
        scheduler.step(val_metrics["total"])

        for name, value in train_metrics.items():
            writer.add_scalar(f"loss/train_{name}", value, epoch)
        for name, value in val_metrics.items():
            writer.add_scalar(f"loss/validation_{name}", value, epoch)
        writer.add_scalar("training/beta", beta, epoch)
        writer.add_scalar(
            "training/learning_rate", optimizer.param_groups[0]["lr"], epoch
        )

        if epoch == 1 or epoch % args.image_log_interval == 0:
            log_images(writer, model, val_loader, device, epoch)

        is_best = val_metrics["total"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["total"]
        save_checkpoint(
            args.output_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            best_val_loss,
            args,
        )
        if is_best:
            save_checkpoint(
                args.output_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_loss,
                args,
            )

        print(
            f"Epoch {epoch:03d}/{args.epochs} | beta {beta:.5f} | "
            f"train {train_metrics['total']:.4f} | "
            f"validation {val_metrics['total']:.4f}"
        )
        writer.flush()

    writer.close()
    print(f"Best checkpoint: {args.output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
