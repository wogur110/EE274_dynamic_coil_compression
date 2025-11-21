import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.serialization import add_safe_globals

add_safe_globals([np._core.multiarray.scalar])


def load_results(path):
    data = torch.load(path)
    # Convert tensors to lists if needed
    def to_list(x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy().tolist()
        return list(x)
    return {
        'bpp': to_list(data.get('bpp', [])),
        'psnr': to_list(data.get('psnr', [])),
        'label': data.get('label', None)
    }


def main():
    base = "results"
    configs = [
        ("JPEG (refer1a)", os.path.join(base, "refer1a_jpeg_compression", "results_jpeg.pt"), "tab:blue"),
        ("DCT (refer1b)", os.path.join(base, "refer1b_dct_compression", "results_dct.pt"), "tab:orange"),
        ("VD Poisson (refer1c)", os.path.join(base, "refer1c_fft_compression", "results_fft.pt"), "tab:green"),
        ("PCA (Uniform)", os.path.join(base, "uniform_coil_compression", "results_pca.pt"), "tab:red"),
    ]

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for label, path, color in configs:
        if not os.path.exists(path):
            print(f"Skipping {label}: {path} not found")
            continue
        data = torch.load(path, weights_only=False)
        bpp = data.get('bpp', [])
        psnr = data.get('psnr', [])
        ssim = data.get('ssim', [])
        if isinstance(bpp, torch.Tensor):
            bpp = bpp.cpu().numpy()
        if isinstance(psnr, torch.Tensor):
            psnr = psnr.cpu().numpy()
        if isinstance(ssim, torch.Tensor):
            ssim = ssim.cpu().numpy()
        bpp = list(bpp)
        psnr = list(psnr)
        ssim = list(ssim)
        if len(bpp) == 0:
            continue
        # sort by bpp for nicer curve
        pairs_psnr = sorted(zip(bpp, psnr))
        pairs_ssim = sorted(zip(bpp, ssim))
        bpp_sorted_psnr, psnr_sorted = zip(*pairs_psnr)
        bpp_sorted_ssim, ssim_sorted = zip(*pairs_ssim)
        
        # Plot PSNR on left subplot
        ax1.plot(bpp_sorted_psnr, psnr_sorted, marker='o', color=color, label=label, linewidth=2, markersize=4)
        # Plot SSIM on right subplot
        ax2.plot(bpp_sorted_ssim, ssim_sorted, marker='o', color=color, label=label, linewidth=2, markersize=4)

    # Configure PSNR plot
    ax1.set_xlabel("Bits per complex coil pixel (bpp)", fontsize=12)
    ax1.set_ylabel("PSNR (dB)", fontsize=12)
    ax1.set_title("Reference Methods: Rate–Distortion (PSNR)", fontsize=14, fontweight='bold')
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend(loc='best', fontsize=10)
    
    # Configure SSIM plot
    ax2.set_xlabel("Bits per complex coil pixel (bpp)", fontsize=12)
    ax2.set_ylabel("SSIM", fontsize=12)
    ax2.set_title("Reference Methods: Rate–Distortion (SSIM)", fontsize=14, fontweight='bold')
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.legend(loc='best', fontsize=10)

    plt.tight_layout()
    
    out_dir = base
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "rd_curve_references.png")
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved combined RD curve to {out_path}")


if __name__ == "__main__":
    main()

