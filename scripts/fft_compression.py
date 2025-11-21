import torch
import numpy as np
import sigpy as sp
import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
from project_JH.utils.mri_utils import load_imgs, run_espirit_pipeline, DATA_DIR, quantize_and_encode

def fft_compress_decompress(imgs, keep_percent, quant_bits=8):
    """
    Per-coil FFT compression.
    imgs: (N, H, W)
    keep_percent: float (0 to 1), fraction of coefficients to keep (center crop).
    """
    N, H, W = imgs.shape
    
    # 1. FFT
    ksp = sp.fft(imgs, axes=(-2, -1))
    
    # 2. Truncation (Low-pass)
    keep_ratio = np.sqrt(keep_percent)
    kh = int(H * keep_ratio)
    kw = int(W * keep_ratio)
    
    # Ensure even
    if kh % 2 != 0: kh += 1
    if kw % 2 != 0: kw += 1
    
    cy, cx = H // 2, W // 2
    sy = cy - kh // 2
    sx = cx - kw // 2
    
    # Extract center
    ksp_center = ksp[..., sy:sy+kh, sx:sx+kw]
    
    # 3. Quantize and Encode
    ksp_rec_center, bits = quantize_and_encode(ksp_center, bits=quant_bits)
    
    # 4. Recon (Pad with zeros)
    ksp_rec = np.zeros_like(ksp)
    ksp_rec[..., sy:sy+kh, sx:sx+kw] = ksp_rec_center
    
    # 5. IFFT
    imgs_rec = sp.ifft(ksp_rec, axes=(-2, -1))
    
    return imgs_rec, bits

def run_fft_experiment():
    # Load
    imgs = load_imgs()
    ref_img_path = os.path.join("results", "ref_image.pt")
    print(f"Loading reference from {ref_img_path}...")
    try:
        ref_img = torch.load(ref_img_path).numpy()
    except:
         # Fallback if reference not generated yet
         print("Reference image not found. Please run generate_reference.py first.")
         return
    
    N, H, W = imgs.shape
    num_pixels = H * W
    
    # Sweep parameters
    # Vary amount of coefficients kept
    keep_percents = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
    quant_bits = 8 
    
    results = {'bpp': [], 'psnr': [], 'ssim': []}
    
    print("Starting FFT compression sweep...")
    
    for kp in keep_percents:
        print(f"\n--- Keep: {kp*100:.1f}% ---")
        
        rec_imgs, bits = fft_compress_decompress(imgs, kp, quant_bits)
        bpp = bits / num_pixels
        
        p, s = run_espirit_pipeline(rec_imgs, ref_img, verbose=False)
        
        results['bpp'].append(bpp)
        results['psnr'].append(p)
        results['ssim'].append(s)
        
        print(f"BPP: {bpp:.2f}, PSNR: {p:.2f} dB, SSIM: {s:.4f}")
        
    # Save
    torch.save(results, os.path.join("results", "results_fft.pt"))
    
    # Plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(results['bpp'], results['psnr'], 'o-')
    plt.xlabel('Bits per Pixel (bpp)')
    plt.ylabel('PSNR (dB)')
    plt.title('FFT: Rate-Distortion (PSNR)')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(results['bpp'], results['ssim'], 'o-')
    plt.xlabel('Bits per Pixel (bpp)')
    plt.ylabel('SSIM')
    plt.title('FFT: Rate-Distortion (SSIM)')
    plt.grid(True)
    
    plt.savefig(os.path.join("results", "fft_rd_curve.png"))
    print("Done.")

if __name__ == "__main__":
    run_fft_experiment()
