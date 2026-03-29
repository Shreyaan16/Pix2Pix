import argparse
import csv
import math
import os
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

import config
from dataset import MapDataset
from generator_model import Generator


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Convert tensors from [-1, 1] to [0, 1]."""
    return (tensor * 0.5 + 0.5).clamp(0.0, 1.0)


def resolve_eval_dir() -> str:
    """Pick the first existing validation directory."""
    candidates = [
        config.VAL_DIR,
        os.path.join("maps", "val"),
        os.path.join("data", "maps", "val"),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(
        "Could not find a validation directory. Checked: " + ", ".join(candidates)
    )


def list_generator_checkpoints(models_dir: str) -> List[str]:
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    checkpoints = []
    for name in sorted(os.listdir(models_dir)):
        lower = name.lower()
        if lower.startswith("gen") and (lower.endswith(".pth") or lower.endswith(".pth.tar")):
            checkpoints.append(os.path.join(models_dir, name))

    if not checkpoints:
        raise FileNotFoundError(
            f"No generator checkpoints found in {models_dir}. Expected names like gen*.pth.tar"
        )
    return checkpoints


def load_generator_from_checkpoint(checkpoint_path: str, device: str) -> Generator:
    gen = Generator(in_channels=3, features=64).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    gen.load_state_dict(state_dict)
    gen.eval()
    return gen


def _gaussian_kernel(window_size: int, sigma: float, channels: int, device: str) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32, device=device)
    coords -= window_size // 2
    gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    kernel_2d = gauss[:, None] * gauss[None, :]
    kernel_2d = kernel_2d.expand(channels, 1, window_size, window_size).contiguous()
    return kernel_2d


def ssim_batch(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> torch.Tensor:
    """Compute SSIM per sample for NCHW images in [0, 1]."""
    channels = pred.size(1)
    window = _gaussian_kernel(window_size, sigma, channels, pred.device)
    padding = window_size // 2

    mu_x = F.conv2d(pred, window, padding=padding, groups=channels)
    mu_y = F.conv2d(target, window, padding=padding, groups=channels)

    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(pred * pred, window, padding=padding, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(target * target, window, padding=padding, groups=channels) - mu_y2
    sigma_xy = F.conv2d(pred * target, window, padding=padding, groups=channels) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = numerator / (denominator + 1e-12)
    return ssim_map.mean(dim=(1, 2, 3))


def compute_metrics(pred_01: torch.Tensor, target_01: torch.Tensor) -> Dict[str, float]:
    mse_per_sample = ((pred_01 - target_01) ** 2).mean(dim=(1, 2, 3))
    mae_per_sample = (pred_01 - target_01).abs().mean(dim=(1, 2, 3))
    rmse_per_sample = torch.sqrt(mse_per_sample + 1e-12)
    ssim_per_sample = ssim_batch(pred_01, target_01)
    psnr_per_sample = 10.0 * torch.log10(1.0 / (mse_per_sample + 1e-12))

    return {
        "SSIM": float(ssim_per_sample.mean().item()),
        "PSNR": float(psnr_per_sample.mean().item()),
        "MAE": float(mae_per_sample.mean().item()),
        "MSE": float(mse_per_sample.mean().item()),
        "RMSE": float(rmse_per_sample.mean().item()),
    }


def save_prediction_visuals(
    model_name: str,
    out_root: str,
    index: int,
    x_01: torch.Tensor,
    y_fake_01: torch.Tensor,
    y_real_01: torch.Tensor,
) -> None:
    model_dir = os.path.join(out_root, model_name, "predictions")
    os.makedirs(model_dir, exist_ok=True)

    error_map = (y_fake_01 - y_real_01).abs()
    panel = torch.stack([x_01, y_fake_01, y_real_01, error_map], dim=0)
    save_image(panel, os.path.join(model_dir, f"sample_{index:04d}.png"), nrow=4)


def save_activation_grid(
    activation: torch.Tensor,
    out_file: str,
    max_channels: int,
) -> None:
    """Save first channels of a layer activation as a grid image."""
    if activation.dim() != 4:
        return

    feature_maps = activation[0]
    channels = min(feature_maps.size(0), max_channels)
    feature_maps = feature_maps[:channels]

    # Per-channel min-max normalization for visualization stability.
    channel_min = feature_maps.amin(dim=(1, 2), keepdim=True)
    channel_max = feature_maps.amax(dim=(1, 2), keepdim=True)
    feature_maps = (feature_maps - channel_min) / (channel_max - channel_min + 1e-8)

    feature_maps = feature_maps.unsqueeze(1)  # N,1,H,W
    nrow = max(1, int(math.sqrt(channels)))
    grid = make_grid(feature_maps, nrow=nrow, padding=2)
    save_image(grid, out_file)


def save_layerwise_visuals(
    gen: Generator,
    x: torch.Tensor,
    out_root: str,
    model_name: str,
    max_channels: int,
) -> None:
    out_dir = os.path.join(out_root, model_name, "layers")
    os.makedirs(out_dir, exist_ok=True)

    activations: Dict[str, torch.Tensor] = {}
    hooks = []

    def _capture_activation(layer_name: str):
        def _hook(_module, _inputs, output):
            if layer_name not in activations:
                activations[layer_name] = output.detach().cpu()
            return None

        return _hook

    for name, module in gen.named_modules():
        if name == "":
            continue
        if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            hooks.append(module.register_forward_hook(_capture_activation(name)))

    with torch.no_grad():
        _ = gen(x)

    for hook in hooks:
        hook.remove()

    for layer_name, act in activations.items():
        safe_name = layer_name.replace(".", "_")
        out_file = os.path.join(out_dir, f"{safe_name}.png")
        save_activation_grid(act, out_file, max_channels=max_channels)


def evaluate_model(
    checkpoint_path: str,
    loader: DataLoader,
    device: str,
    out_root: str,
    max_samples: int,
    num_visualizations: int,
    activation_channels: int,
) -> Dict[str, float]:
    model_base = os.path.basename(checkpoint_path)
    if model_base.endswith(".pth.tar"):
        model_name = model_base[: -len(".pth.tar")]
    else:
        model_name = os.path.splitext(model_base)[0]
    gen = load_generator_from_checkpoint(checkpoint_path, device)

    metric_totals = {
        "SSIM": 0.0,
        "PSNR": 0.0,
        "MAE": 0.0,
        "MSE": 0.0,
        "RMSE": 0.0,
    }
    seen = 0
    saved_visuals = 0
    saved_layers = False

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            y_fake = gen(x)

        x_01 = denormalize(x)
        y_fake_01 = denormalize(y_fake)
        y_01 = denormalize(y)

        metrics = compute_metrics(y_fake_01, y_01)
        batch_size = x.size(0)
        for key in metric_totals:
            metric_totals[key] += metrics[key] * batch_size
        seen += batch_size

        if not saved_layers:
            save_layerwise_visuals(
                gen=gen,
                x=x[:1],
                out_root=out_root,
                model_name=model_name,
                max_channels=activation_channels,
            )
            saved_layers = True

        if saved_visuals < num_visualizations:
            for i in range(batch_size):
                if saved_visuals >= num_visualizations:
                    break
                global_index = batch_idx * loader.batch_size + i
                save_prediction_visuals(
                    model_name=model_name,
                    out_root=out_root,
                    index=global_index,
                    x_01=x_01[i],
                    y_fake_01=y_fake_01[i],
                    y_real_01=y_01[i],
                )
                saved_visuals += 1

        if seen >= max_samples:
            break

    if seen == 0:
        raise RuntimeError("No samples were evaluated. Check dataset path and dataloader settings.")

    return {key: value / seen for key, value in metric_totals.items()}


def save_summary_csv(rows: List[Dict[str, float]], out_file: str) -> None:
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    fieldnames = ["model", "SSIM", "PSNR", "MAE", "MSE", "RMSE"]
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pix2pix generator checkpoints.")
    parser.add_argument("--models-dir", type=str, default="goodmodels", help="Directory with generator checkpoints.")
    parser.add_argument("--model-path", type=str, default="", help="Optional single generator checkpoint path.")
    parser.add_argument("--data-dir", type=str, default="", help="Validation dataset directory.")
    parser.add_argument("--output-dir", type=str, default="evaluation", help="Output directory for images and metrics.")
    parser.add_argument("--batch-size", type=int, default=1, help="Evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader workers.")
    parser.add_argument("--max-samples", type=int, default=200, help="Maximum number of samples per model.")
    parser.add_argument("--num-visualizations", type=int, default=20, help="Number of prediction panels to save per model.")
    parser.add_argument("--activation-channels", type=int, default=16, help="Max channels visualized per layer activation image.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = config.DEVICE

    data_dir = args.data_dir if args.data_dir else resolve_eval_dir()
    dataset = MapDataset(root_dir=data_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    if args.model_path:
        checkpoints = [args.model_path]
    else:
        checkpoints = list_generator_checkpoints(args.models_dir)

    rows = []
    print(f"Evaluating {len(checkpoints)} model(s) on {data_dir} using device={device}")

    for checkpoint in checkpoints:
        print(f"\nEvaluating: {checkpoint}")
        metrics = evaluate_model(
            checkpoint_path=checkpoint,
            loader=loader,
            device=device,
            out_root=args.output_dir,
            max_samples=args.max_samples,
            num_visualizations=args.num_visualizations,
            activation_channels=args.activation_channels,
        )
        row = {
            "model": os.path.basename(checkpoint),
            "SSIM": round(metrics["SSIM"], 6),
            "PSNR": round(metrics["PSNR"], 6),
            "MAE": round(metrics["MAE"], 6),
            "MSE": round(metrics["MSE"], 6),
            "RMSE": round(metrics["RMSE"], 6),
        }
        rows.append(row)
        print(
            "Metrics -> "
            f"SSIM: {row['SSIM']}, "
            f"PSNR: {row['PSNR']}, "
            f"MAE: {row['MAE']}, "
            f"MSE: {row['MSE']}, "
            f"RMSE: {row['RMSE']}"
        )

    summary_csv = os.path.join(args.output_dir, "metrics_summary.csv")
    save_summary_csv(rows, summary_csv)
    print(f"\nSaved summary: {summary_csv}")


if __name__ == "__main__":
    main()