#!/usr/bin/env python3
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _import_or_exit():
    try:
        import lpips  # noqa: F401
    except Exception:
        print("Missing dependency: lpips", file=sys.stderr)
        print("Install with: pip install lpips", file=sys.stderr)
        sys.exit(1)

    try:
        import open_clip  # noqa: F401
        return "open_clip"
    except Exception:
        try:
            import clip  # noqa: F401
            return "clip"
        except Exception:
            print("Missing CLIP backend.", file=sys.stderr)
            print("Install with: pip install open_clip_torch", file=sys.stderr)
            sys.exit(1)


def _get_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def to_lpips_tensor(img: Image.Image, device: str) -> torch.Tensor:
    arr = np.array(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    t = t * 2.0 - 1.0
    return t.to(device)


@torch.no_grad()
def clip_similarity(clip_model, clip_preprocess, img_a: Image.Image, img_b: Image.Image, device: str) -> float:
    a = clip_preprocess(img_a).unsqueeze(0).to(device)
    b = clip_preprocess(img_b).unsqueeze(0).to(device)
    fa = clip_model.encode_image(a)
    fb = clip_model.encode_image(b)
    fa = fa / fa.norm(dim=-1, keepdim=True)
    fb = fb / fb.norm(dim=-1, keepdim=True)
    return (fa * fb).sum(dim=-1).item()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Compute LPIPS/PSNR/CLIP/SSIM for gt vs pred composite.png.")
    parser.add_argument(
        "--gt-dir",
        default=os.path.join(script_dir, "gt"),
        help="Path to gt directory containing <id>/composite.png",
    )
    parser.add_argument(
        "--pred-dir",
        default=os.path.join(script_dir, "pred"),
        help="Path to pred directory containing <id>/composite.png",
    )
    parser.add_argument(
        "--out",
        default="metrics2.json",
        help="Output JSON path (default: metrics2.json)",
    )
    args = parser.parse_args()

    clip_backend = _import_or_exit()

    import lpips

    device = _get_device()

    lpips_model = lpips.LPIPS(net="alex").to(device).eval()

    if clip_backend == "open_clip":
        import open_clip

        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        clip_model = clip_model.to(device).eval()
    else:
        import clip

        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
        clip_model.eval()

    gt_paths = glob.glob(os.path.join(args.gt_dir, "*", "composite.png"))
    pred_paths = glob.glob(os.path.join(args.pred_dir, "*", "composite.png"))

    gt_map = {os.path.basename(os.path.dirname(p)): p for p in gt_paths}
    pred_map = {os.path.basename(os.path.dirname(p)): p for p in pred_paths}

    common_ids = sorted(set(gt_map) & set(pred_map))
    if not common_ids:
        raise FileNotFoundError("No matching composite.png pairs found.")

    rows = []
    for id_ in common_ids:
        gt_img = Image.open(gt_map[id_]).convert("RGB")
        pred_img = Image.open(pred_map[id_]).convert("RGB")

        if pred_img.size != gt_img.size:
            pred_img = pred_img.resize(gt_img.size, Image.BICUBIC)

        gt_np = np.array(gt_img).astype(np.float32) / 255.0
        pred_np = np.array(pred_img).astype(np.float32) / 255.0

        psnr = peak_signal_noise_ratio(gt_np, pred_np, data_range=1.0)
        ssim = structural_similarity(gt_np, pred_np, channel_axis=2, data_range=1.0)

        lp = lpips_model(to_lpips_tensor(gt_img, device), to_lpips_tensor(pred_img, device)).item()
        clip_score = clip_similarity(clip_model, clip_preprocess, gt_img, pred_img, device)

        rows.append(
            {
                "id": id_,
                "lpips": float(lp),
                "psnr": float(psnr),
                "clip_score": float(clip_score),
                "ssim": float(ssim),
            }
        )

    metrics = {
        "rows": rows,
        "mean": {
            "lpips": float(np.mean([r["lpips"] for r in rows])),
            "psnr": float(np.mean([r["psnr"] for r in rows])),
            "clip_score": float(np.mean([r["clip_score"] for r in rows])),
            "ssim": float(np.mean([r["ssim"] for r in rows])),
        },
        "count": len(rows),
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved: {args.out}")
    print(f"Count: {metrics['count']}")
    print("Mean:", metrics["mean"])


if __name__ == "__main__":
    main()
