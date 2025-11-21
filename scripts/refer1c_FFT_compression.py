import torch
import numpy as np
import sigpy as sp
import os
import sys
import zlib
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
from project_JH.utils.mri_utils import (
    load_imgs,
    run_espirit_pipeline,
    DATA_DIR,
    get_poisson_mask,
    normalize_complex_image,
    get_device,
    crop_center,
    quantize_and_encode,
)
from project_JH.utils.plot_utils import save_rd_curve
from project_JH.utils.espirit_torch import csm_from_espirit

def run_undersampling_experiment():
    # Load data
    imgs = load_imgs()
    ref_img_path = os.path.join("results", "refer0", "ref_image.pt")
    try:
        ref_img = torch.load(ref_img_path).numpy()
    except:
        print("Reference image not found.")
        return
         
    N, H, W = imgs.shape
    num_pixels = H * W
    num_pixels_total = N * num_pixels
    
    # Convert to k-space (Ground Truth)
    ksp_gt = sp.fft(imgs, axes=(-2, -1))
    ksp_gt_flat = ksp_gt.reshape(N, -1)
    
    # Sweep accelerations
    accelerations = [1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14]
    quant_bits = 9
    
    results = {'accel': [], 'bpp': [], 'psnr': [], 'ssim': []}
    
    output_dir = os.path.join("results", "refer1c_fft_compression")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting Phase 1B: Undersampling (Poisson Disc) sweep...")
    
    for R in accelerations:
        print(f"\n--- Acceleration: {R}x ---")
        
        # Generate mask
        # ACS region 32x32 is passed as calib
        # For acceleration 1.0 (full sampling), create full mask
        if R == 1.0:
            mask = np.ones((H, W), dtype=np.float32)
        else:
            mask = get_poisson_mask((H, W), R, calib=(32, 32), seed=42)
        
        # Apply mask
        mask_arr = mask.astype(bool)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[0]
        mask_flat = mask_arr.reshape(-1)
        num_samples = np.sum(mask_flat)
        
        # Gather sampled k-space values and quantize/compress
        samples = ksp_gt_flat[:, mask_flat]
        rec_samples, bits_coeffs = quantize_and_encode(samples, bits=quant_bits)
        
        # Encode mask (shared across coils)
        mask_bytes = np.packbits(mask_arr.astype(np.uint8))
        mask_compressed = zlib.compress(mask_bytes.tobytes())
        bits_mask = len(mask_compressed) * 8 + 2 * 32  # include dims overhead
        
        total_bits = bits_coeffs + bits_mask
        bpp = total_bits / num_pixels_total
        
        # Reconstruct quantized k-space
        ksp_rec_flat = np.zeros_like(ksp_gt_flat, dtype=samples.dtype)
        ksp_rec_flat[:, mask_flat] = rec_samples
        ksp_rec = ksp_rec_flat.reshape(N, H, W)
        
        # IFFT to coil images
        imgs_rec = sp.ifft(ksp_rec, axes=(-2, -1))
        
        # Evaluation Pipeline
        # This will run ESPIRiT on the aliased images (zero-filled).
        p, s = run_espirit_pipeline(imgs_rec, ref_img, verbose=False)
        
        results['accel'].append(R)
        results['bpp'].append(bpp)
        results['psnr'].append(p)
        results['ssim'].append(s)
        
        print(f"Accel: {R}x, BPP: {bpp:.2f}, PSNR: {p:.2f} dB, SSIM: {s:.4f}")
        
        # Reconstruct final image for saving
        ksp_rec = sp.fft(imgs_rec, axes=(-2, -1))
        ksp_rec_torch = torch.from_numpy(ksp_rec).to(get_device())
        ksp_cal = crop_center(ksp_rec_torch, 32)
        im_size = imgs_rec.shape[-2:]
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
        rec_final = np.sum(imgs_rec * np.conj(maps), axis=0) / weights
        rec_final = normalize_complex_image(rec_final)
        
        rec_path = os.path.join(output_dir, f"rec_img_accel{R}.pt")
        torch.save(torch.from_numpy(rec_final), rec_path)
        
        plt.figure(figsize=(5, 5))
        plt.imshow(np.abs(rec_final), cmap='gray')
        plt.title(f"VD Poisson R={R}\nBPP={bpp:.1f}, PSNR={p:.2f}dB, SSIM={s:.4f}")
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f"rec_img_accel{R}.png"), bbox_inches='tight')
        plt.close()
        
    torch.save(results, os.path.join(output_dir, "results_fft.pt"))
    
    save_rd_curve(results, "VD Poisson FFT", "fft_rd_curve.png", output_dir=output_dir)
    print("Done.")

if __name__ == "__main__":
    run_undersampling_experiment()

