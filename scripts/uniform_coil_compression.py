import torch
import numpy as np
import sigpy as sp
import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
from project_JH.utils.mri_utils import (
    load_imgs,
    run_espirit_pipeline,
    DATA_DIR,
    quantize_and_encode,
    normalize_complex_image,
    get_device,
    crop_center,
)
from project_JH.utils.plot_utils import save_rd_curve
from project_JH.utils.espirit_torch import csm_from_espirit

def pca_compress_decompress(imgs, K_pca, quant_bits=8):
    """
    Typical PCA compression in image domain.
    imgs: (N, H, W)
    K_pca: Number of components to keep.
    """
    N, H, W = imgs.shape
    
    # V = rearrange(imgs, 'n h w -> n (h w)')
    V = imgs.reshape(N, -1)
    
    # 2. Covariance
    # Center data? Usually PCA centers. But MRI coils often assume zero mean noise, but signal is not zero mean.
    # Standard coil compression usually doesn't center signal relative to pixel mean?
    # "PCA on this covariance yields a global coil basis."
    # Usually C = V @ V.H (if centered) or just correlation matrix.
    # Let's assume standard EVD on V @ V^H.
    
    # Efficient calculation: V is (64, 90k). V @ V.H is (64, 64).
    C = V @ V.conj().T
    
    # 3. Eigen decomposition
    # Hermitian.
    # eigh returns eigenvalues in ascending order.
    vals, vecs = np.linalg.eigh(C)
    
    # Sort descending
    idx = np.argsort(vals)[::-1]
    vecs = vecs[:, idx]
    # vals = vals[idx]
    
    # 4. Keep K
    U_k = vecs[:, :K_pca] # (N, K)
    
    # 5. Transform
    # Y = U_k^H @ V  -> (K, N).H @ (N, P) -> (K, P)
    Y = U_k.conj().T @ V
    
    # 6. Quantize
    Y_rec, bits_coeffs = quantize_and_encode(Y, bits=quant_bits)
    
    # Overhead for basis
    # U_k is complex float32/64.
    # We can quantize basis too, but usually sent high precision.
    # 64 * K * 2 (real/imag) * 4 bytes (float32) * 8 bits
    bits_basis = K_pca * N * 2 * 4 * 8
    
    total_bits = bits_coeffs + bits_basis
    
    # 7. Reconstruct
    # V_rec = U_k @ Y_rec
    V_rec = U_k @ Y_rec
    
    # imgs_rec = rearrange(V_rec, 'n (h w) -> n h w', h=H, w=W)
    imgs_rec = V_rec.reshape(N, H, W)
    
    return imgs_rec, total_bits

def run_pca_experiment():
    imgs = load_imgs()
    ref_img_path = os.path.join("results", "refer0", "ref_image.pt")
    print(f"Loading reference from {ref_img_path}...")
    try:
        ref_img = torch.load(ref_img_path).numpy()
    except:
         print("Reference image not found.")
         return
         
    N, H, W = imgs.shape
    num_pixels = H * W
    num_pixels_total = N * num_pixels
    
    # Sweep K
    # N=64.
    Ks = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32]
    quant_bits = 8
    
    results = {'bpp': [], 'psnr': [], 'ssim': [], 'rank': []}
    output_dir = os.path.join("results", "uniform_coil_compression")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting PCA compression sweep...")
    
    for K in Ks:
        print(f"\n--- K: {K} ---")
        
        rec_imgs, bits = pca_compress_decompress(imgs, K, quant_bits)
        bpp = bits / num_pixels_total
        
        p, s = run_espirit_pipeline(rec_imgs, ref_img, verbose=False)
        
        results['bpp'].append(bpp)
        results['psnr'].append(p)
        results['ssim'].append(s)
        results['rank'].append(K)
        
        print(f"BPP: {bpp:.2f}, PSNR: {p:.2f} dB, SSIM: {s:.4f}")
        
        # Reconstruct final ESPIRiT image for saving
        ksp_rec = sp.fft(rec_imgs, axes=(-2, -1))
        ksp_rec_torch = torch.from_numpy(ksp_rec).to(get_device())
        ksp_cal = crop_center(ksp_rec_torch, 32)
        im_size = rec_imgs.shape[-2:]
        maps, _ = csm_from_espirit(
            ksp_cal,
            im_size=im_size,
            thresh=0.02,
            kernel_width=6,
            crp=None,
            max_iter=30,
            verbose=False
        )
        if isinstance(maps, torch.Tensor):
            maps = maps.cpu().numpy()
        weights = np.sum(np.abs(maps)**2, axis=0) + 1e-16
        rec_final = np.sum(rec_imgs * np.conj(maps), axis=0) / weights
        rec_final = normalize_complex_image(rec_final)
        
        rec_path = os.path.join(output_dir, f"rec_img_rank{K}.pt")
        torch.save(torch.from_numpy(rec_final), rec_path)
        
        plt.figure(figsize=(5, 5))
        plt.imshow(np.abs(rec_final), cmap='gray')
        plt.title(f"PCA Rank={K}\nBPP={bpp:.2f}, PSNR={p:.2f}dB, SSIM={s:.4f}")
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f"rec_img_rank{K}.png"), bbox_inches='tight')
        plt.close()
        
    torch.save(results, os.path.join(output_dir, "results_pca.pt"))
    
    save_rd_curve(results, "PCA (Uniform)", "pca_rd_curve.png", output_dir=output_dir)
    print("Done.")

if __name__ == "__main__":
    run_pca_experiment()

