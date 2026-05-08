from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_msssim import ssim
from torch.utils.data import DataLoader, Dataset

from backend.inference import UNet


@dataclass(frozen=True)
class Config:
    data_dir: Path
    pretrained_model_path: Path
    results_dir: Path
    epochs: int
    batch_size: int
    lr: float
    subset_size: int
    eval_images: int
    image_size: int
    train_noise_levels: tuple[float, ...]
    test_noise_levels: tuple[float, ...]
    seed: int


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="One-day novelty pipeline for U-Net denoising research.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/images"))
    parser.add_argument("--pretrained-model", type=Path, default=Path("experiments/notebook_run/models/unet_model.pth"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--subset-size", type=int, default=240)
    parser.add_argument("--eval-images", type=int, default=24)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--train-noise-levels", type=float, nargs="+", default=[0.2, 0.4, 0.6])
    parser.add_argument("--test-noise-levels", type=float, nargs="+", default=[0.2, 0.4, 0.6])
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    return Config(
        data_dir=args.data_dir,
        pretrained_model_path=args.pretrained_model,
        results_dir=args.results_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        subset_size=args.subset_size,
        eval_images=args.eval_images,
        image_size=args.image_size,
        train_noise_levels=tuple(args.train_noise_levels),
        test_noise_levels=tuple(args.test_noise_levels),
        seed=args.seed,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_speckle_noise(img: torch.Tensor, var: float) -> torch.Tensor:
    noise = torch.randn_like(img) * var
    return torch.clamp(img + img * noise, 0.0, 1.0)


def hybrid_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(pred, target)
    ssim_loss = 1 - ssim(pred, target, data_range=1.0, size_average=True)
    return mse + 0.5 * ssim_loss


def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> float:
    mse = F.mse_loss(pred, target).item()
    mse = max(mse, eps)
    return float(10.0 * np.log10(1.0 / mse))


def ssim_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(ssim(pred, target, data_range=1.0, size_average=True).item())


def load_gray_tensor(path: Path, image_size: int) -> torch.Tensor:
    img = Image.open(path).convert("L").resize((image_size, image_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return tensor


def image_paths(data_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return [
        p
        for p in sorted(data_dir.iterdir())
        if p.suffix.lower() in exts and "mask" not in p.name.lower()
    ]


class SpeckleDataset(Dataset):
    def __init__(self, paths: list[Path], image_size: int, noise_level: float) -> None:
        self.paths = paths
        self.image_size = image_size
        self.noise_level = noise_level

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        clean = load_gray_tensor(self.paths[idx], self.image_size).squeeze(0)
        noisy = add_speckle_noise(clean.unsqueeze(0), self.noise_level).squeeze(0)
        return noisy, clean


def choose_pretrained_model(path: Path) -> Path:
    if path.exists():
        return path
    fallback = Path("backend/models/unet_model.pth")
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No pretrained model found at {path} or {fallback}")


def maybe_fallback_image_size(cfg: Config, device: torch.device) -> int:
    if device.type != "cuda":
        return cfg.image_size
    model = UNet().to(device).eval()
    try:
        x = torch.randn(1, 1, cfg.image_size, cfg.image_size, device=device)
        with torch.no_grad():
            _ = model(x)
        return cfg.image_size
    except RuntimeError:
        torch.cuda.empty_cache()
        return 256


def save_image_grid(clean: torch.Tensor, noisy: torch.Tensor, pred: torch.Tensor, path: Path) -> None:
    clean_np = clean.squeeze().detach().cpu().numpy()
    noisy_np = noisy.squeeze().detach().cpu().numpy()
    pred_np = pred.squeeze().detach().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(clean_np, cmap="gray")
    axes[0].set_title("Input clean")
    axes[1].imshow(noisy_np, cmap="gray")
    axes[1].set_title("Noisy")
    axes[2].imshow(pred_np, cmap="gray")
    axes[2].set_title("Output")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def evaluate_model(
    model: UNet,
    paths: Iterable[Path],
    image_size: int,
    noise_level: float,
    device: torch.device,
) -> tuple[list[dict[str, float | str]], float, float]:
    rows: list[dict[str, float | str]] = []
    model.eval()
    with torch.no_grad():
        for p in paths:
            clean = load_gray_tensor(p, image_size).to(device)
            noisy = add_speckle_noise(clean, noise_level)
            pred = model(noisy)
            rows.append(
                {
                    "image": p.name,
                    "noise": noise_level,
                    "psnr": psnr(pred, clean),
                    "ssim": ssim_score(pred, clean),
                }
            )
    mean_psnr = float(np.mean([r["psnr"] for r in rows])) if rows else 0.0
    mean_ssim = float(np.mean([r["ssim"] for r in rows])) if rows else 0.0
    return rows, mean_psnr, mean_ssim


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = maybe_fallback_image_size(cfg, device)

    results_images = cfg.results_dir / "images"
    results_plots = cfg.results_dir / "plots"
    results_tables = cfg.results_dir / "tables"
    for d in (results_images, results_plots, results_tables):
        d.mkdir(parents=True, exist_ok=True)

    model_path = choose_pretrained_model(cfg.pretrained_model_path)
    all_paths = image_paths(cfg.data_dir)
    if len(all_paths) < 40:
        raise RuntimeError(f"Need at least 40 images for this pipeline. Found: {len(all_paths)}")

    eval_paths = all_paths[: min(cfg.eval_images, len(all_paths) // 4)]
    train_candidates = all_paths[min(cfg.eval_images, len(all_paths) // 4) :]
    train_paths = train_candidates[: min(cfg.subset_size, len(train_candidates))]

    base_model = UNet().to(device)
    base_state = torch.load(model_path, map_location=device)
    base_model.load_state_dict(base_state, strict=False)

    # Phase 1: baseline metrics (MSE-trained model inference under moderate noise)
    baseline_rows, baseline_psnr, baseline_ssim = evaluate_model(
        base_model, eval_paths, image_size=image_size, noise_level=0.4, device=device
    )
    baseline_rows.append({"image": "MEAN", "noise": 0.4, "psnr": baseline_psnr, "ssim": baseline_ssim})
    baseline_csv = results_tables / "baseline_metrics.csv"
    write_csv(baseline_csv, baseline_rows, ["image", "noise", "psnr", "ssim"])

    # Save a representative baseline input/output image.
    sample_clean = load_gray_tensor(eval_paths[0], image_size).to(device)
    sample_noisy = add_speckle_noise(sample_clean, 0.4)
    with torch.no_grad():
        sample_pred = base_model(sample_noisy)
    save_image_grid(sample_clean, sample_noisy, sample_pred, results_images / "before_after.png")

    # Phase 4+5: quick hybrid fine-tune and cross-noise robustness.
    robustness_rows: list[dict[str, float | str]] = []
    comparison_rows: list[dict[str, str | float]] = [
        {"Method": "SVD", "PSNR": float("nan"), "SSIM": float("nan")},
        {"Method": "U-Net (MSE)", "PSNR": baseline_psnr, "SSIM": baseline_ssim},
    ]

    best_hybrid_psnr = -1.0
    best_hybrid_ssim = -1.0
    best_hybrid_state = None
    best_train_noise = None

    for train_noise in cfg.train_noise_levels:
        model = UNet().to(device)
        model.load_state_dict(base_state, strict=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        use_amp = device.type == "cuda"

        train_ds = SpeckleDataset(train_paths, image_size=image_size, noise_level=float(train_noise))
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

        for _ in range(cfg.epochs):
            model.train()
            for noisy, clean in train_loader:
                noisy = noisy.to(device, non_blocking=True)
                clean = clean.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        pred = model(noisy)
                        loss = hybrid_loss(pred, clean)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pred = model(noisy)
                    loss = hybrid_loss(pred, clean)
                    loss.backward()
                    optimizer.step()

        for test_noise in cfg.test_noise_levels:
            _, mean_psnr, mean_ssim = evaluate_model(
                model,
                eval_paths,
                image_size=image_size,
                noise_level=float(test_noise),
                device=device,
            )
            robustness_rows.append(
                {
                    "train_noise": train_noise,
                    "test_noise": test_noise,
                    "psnr": mean_psnr,
                    "ssim": mean_ssim,
                }
            )
            if float(test_noise) == 0.4 and mean_psnr > best_hybrid_psnr:
                best_hybrid_psnr = mean_psnr
                best_hybrid_ssim = mean_ssim
                best_hybrid_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                best_train_noise = train_noise

    if best_hybrid_state is None:
        raise RuntimeError("Hybrid training did not produce a valid model state.")

    hybrid_model_path = cfg.results_dir / "unet_hybrid.pth"
    torch.save(best_hybrid_state, hybrid_model_path)

    robustness_csv = results_tables / "robustness.csv"
    write_csv(robustness_csv, robustness_rows, ["train_noise", "test_noise", "psnr", "ssim"])

    # Plot robustness curves (PSNR vs test noise, one line per train noise).
    plt.figure(figsize=(7, 4))
    for train_noise in sorted(set(float(r["train_noise"]) for r in robustness_rows)):
        xs = [float(r["test_noise"]) for r in robustness_rows if float(r["train_noise"]) == train_noise]
        ys = [float(r["psnr"]) for r in robustness_rows if float(r["train_noise"]) == train_noise]
        order = np.argsort(xs)
        xs_sorted = np.array(xs)[order]
        ys_sorted = np.array(ys)[order]
        plt.plot(xs_sorted, ys_sorted, marker="o", label=f"Train noise {train_noise}")
    plt.xlabel("Noise Level")
    plt.ylabel("PSNR")
    plt.title("Robustness Analysis")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_plots / "robustness.png", dpi=150)
    plt.close()

    # Visual result using best hybrid model at noise=0.4 and a difference map.
    hybrid_model = UNet().to(device)
    hybrid_model.load_state_dict(best_hybrid_state, strict=False)
    hybrid_model.eval()

    clean_vis = load_gray_tensor(eval_paths[0], image_size).to(device)
    noisy_vis = add_speckle_noise(clean_vis, 0.4)
    with torch.no_grad():
        pred_vis = hybrid_model(noisy_vis)
    diff = torch.abs(pred_vis - clean_vis).squeeze().cpu().numpy()

    save_image_grid(clean_vis, noisy_vis, pred_vis, results_images / "before_after.png")
    plt.figure(figsize=(4, 4))
    plt.imshow(diff, cmap="inferno")
    plt.title("Difference map")
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(results_images / "diff_map.png", dpi=150, bbox_inches="tight")
    plt.close()

    comparison_rows.append(
        {
            "Method": "U-Net + Hybrid + Noise",
            "PSNR": best_hybrid_psnr,
            "SSIM": best_hybrid_ssim,
        }
    )
    comparison_csv = results_tables / "comparison_table.csv"
    write_csv(comparison_csv, comparison_rows, ["Method", "PSNR", "SSIM"])

    summary = {
        "device": str(device),
        "image_size": image_size,
        "best_train_noise": best_train_noise,
        "baseline_metrics_csv": str(baseline_csv),
        "robustness_csv": str(robustness_csv),
        "comparison_csv": str(comparison_csv),
        "robustness_plot": str(results_plots / "robustness.png"),
        "before_after": str(results_images / "before_after.png"),
        "diff_map": str(results_images / "diff_map.png"),
        "hybrid_model": str(hybrid_model_path),
    }
    with (cfg.results_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Pipeline complete")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
