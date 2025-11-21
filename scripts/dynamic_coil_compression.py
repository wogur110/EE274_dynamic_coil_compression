import torch
import numpy as np
import sigpy as sp
import os
import sys
import matplotlib.pyplot as plt
# from einops import rearrange

sys.path.append(os.getcwd())
from project_JH.utils.mri_utils import load_imgs, run_espirit_pipeline, DATA_DIR, quantize_and_encode
from project_JH.utils.plot_utils import save_rd_curve

def get_radial_masks(H, W, bands_radii):
    """
    bands_radii: list of floats [0.1, 0.3] relative to max radius.
    Returns list of masks (H, W) boolean.
    """
    cy, cx = H // 2, W // 2
    y = np.arange(H) - cy
    x = np.arange(W) - cx
    Y, X = np.meshgrid(y, x, indexing='ij')
    R = np.sqrt(Y**2 + X**2)
    max_r = np.sqrt(cy**2 + cx**2)
    R_norm = R / max_r
    
    masks = []
    prev_r = 0.0
    for r in bands_radii:
        mask = (R_norm >= prev_r) & (R_norm < r)
        masks.append(mask)
        prev_r = r
    
    # Last band
    mask = (R_norm >= prev_r)
    masks.append(mask)
    
    return masks

def dynamic_fft_pca_compress_decompress(imgs, rank_scale, quant_bits=8):
    """
    Dynamic Band-wise FFT+PCA.
    rank_scale: float (0 to 1), scales the max rank for bands.
    """
    N, H, W = imgs.shape
    
    # 1. FFT
    ksp = sp.fft(imgs, axes=(-2, -1))
    
    # Define bands
    # Low, Mid, High
    bands_radii = [0.15, 0.4] # relative radius
    masks = get_radial_masks(H, W, bands_radii)
    
    # Define ranks per band
    # Max rank is 64.
    # Low: keep most/all. Mid: less. High: few.
    # We scale them by rank_scale.
    # Example: Low=64*scale, Mid=32*scale, High=8*scale
    
    # Ensure integer and at least 1
    def get_rank(base):
        r = int(base * rank_scale)
        return max(1, min(64, r))
        
    ranks = [get_rank(64), get_rank(16), get_rank(4)]
    # Note: High freq often has very little signal, so low rank is fine.
    
    ksp_rec = np.zeros_like(ksp)
    total_bits = 0
    
    for mask, K in zip(masks, ranks):
        # mask is (H, W)
        # Extract coil vectors in this band
        # mask broadcast to (N, H, W)
        
        # Get indices
        # We can flatten and select
        mask_flat = mask.flatten() # (P,)
        num_pixels_band = np.sum(mask_flat)
        
        if num_pixels_band == 0:
            continue
            
        # ksp_flat = rearrange(ksp, 'n h w -> n (h w)')
        ksp_flat = ksp.reshape(N, -1)
        V_band = ksp_flat[:, mask_flat]
        
        # PCA
        C = V_band @ V_band.conj().T
        vals, vecs = np.linalg.eigh(C)
        idx = np.argsort(vals)[::-1]
        vecs = vecs[:, idx]
        
        U_k = vecs[:, :K]
        
        # Transform
        Y = U_k.conj().T @ V_band
        
        # Quantize
        Y_rec, bits_coeffs = quantize_and_encode(Y, bits=quant_bits)
        
        # Overhead
        bits_basis = K * N * 2 * 4 * 8
        total_bits += bits_coeffs + bits_basis
        
        # Reconstruct
        V_band_rec = U_k @ Y_rec
        
        # Place back
        # Create empty flat array for this band (or full array)
        # ksp_rec is (N, H, W)
        # We need to assign back to masked locations
        
        # We can use advanced indexing
        # ksp_rec_flat = rearrange(ksp_rec, 'n h w -> n (h w)') # This is a copy? No, rearrange might be view but usually copy if not contiguous.
        # Better to index directly into (N, H, W) using (N, mask) logic?
        # Or just:
        
        # Assign column by column is slow.
        # Let's use flat view if possible or just reshape at end.
        # But we are accumulating into ksp_rec which is 0 initialized.
        # Since bands are disjoint, we can add.
        
        # Construct full flat array for this band
        V_band_full = np.zeros((N, H*W), dtype=ksp.dtype)
        V_band_full[:, mask_flat] = V_band_rec
        
        # ksp_rec += rearrange(V_band_full, 'n (h w) -> n h w', h=H, w=W)
        ksp_rec += V_band_full.reshape(N, H, W)
        
    # IFFT
    imgs_rec = sp.ifft(ksp_rec, axes=(-2, -1))
    
    return imgs_rec, total_bits

def run_dynamic_experiment():
    imgs = load_imgs()
    ref_img_path = os.path.join("results", "refer0", "ref_image.pt")
    try:
        ref_img = torch.load(ref_img_path).numpy()
    except:
         print("Reference image not found.")
         return
         
    N, H, W = imgs.shape
    num_pixels = H * W
    num_pixels_total = N * num_pixels  # Total complex coil pixels
    
    # Sweep rank scale
    scales = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
    quant_bits = 8
    
    results = {'bpp': [], 'psnr': [], 'ssim': []}
    
    print("Starting Dynamic FFT+PCA sweep...")
    
    for s_val in scales:
        print(f"\n--- Scale: {s_val} ---")
        
        rec_imgs, bits = dynamic_fft_pca_compress_decompress(imgs, s_val, quant_bits)
        bpp = bits / num_pixels_total  # Bits per complex coil pixel
        
        p, s = run_espirit_pipeline(rec_imgs, ref_img, verbose=False)
        
        results['bpp'].append(bpp)
        results['psnr'].append(p)
        results['ssim'].append(s)
        
        print(f"BPP: {bpp:.2f}, PSNR: {p:.2f} dB, SSIM: {s:.4f}")
        
    torch.save(results, os.path.join("results", "results_dynamic.pt"))
    
    save_rd_curve(results, "Dynamic FFT+PCA", "dynamic_rd_curve.png", output_dir="results")
    print("Done.")

if __name__ == "__main__":
    run_dynamic_experiment()

