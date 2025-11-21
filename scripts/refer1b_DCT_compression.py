import torch
import numpy as np
import sigpy as sp
import os
import sys
import matplotlib.pyplot as plt
from scipy.fft import dctn, idctn

# Add current directory to path
sys.path.append(os.getcwd())
from project_JH.utils.mri_utils import (
    load_imgs,
    run_espirit_pipeline,
    DATA_DIR,
    quantize_and_encode,
    normalize_complex_image,
)
from project_JH.utils.plot_utils import save_rd_curve

def dct_compress_decompress(imgs, keep_ratio, quant_bits=8):
    """
    Per-coil 2D DCT compression with water-filling (magnitude thresholding).
    imgs: (N, H, W) complex
    keep_ratio: float (0 to 1), fraction of coefficients to keep (largest magnitude).
    """
    N, H, W = imgs.shape
    num_coeffs = N * H * W
    
    # 1. Separate Real and Imaginary
    # DCT on complex? Usually DCT is on real. 
    # We treat real and imag as separate channels or images.
    
    real = imgs.real
    imag = imgs.imag
    
    # 2. 2D DCT
    # orthonormal DCT (type 2, norm='ortho')
    dct_real = dctn(real, axes=(-2, -1), norm='ortho')
    dct_imag = dctn(imag, axes=(-2, -1), norm='ortho')
    
    # 3. Water-filling / Thresholding
    # "remain the important DCT components"
    # We compute magnitude (or absolute value) and keep the top ones.
    # Should we do it per coil or globally? 
    # "water-filling method" often implies global allocation if optimizing global distortion.
    # If we do it globally across all coils/real/imag, we get better efficiency.
    # Let's do it globally across the whole stack (real+imag).
    
    # Combine coeffs for sorting
    all_coeffs = np.concatenate([dct_real.flatten(), dct_imag.flatten()])
    abs_coeffs = np.abs(all_coeffs)
    
    # Determine threshold
    k = int(len(all_coeffs) * keep_ratio)
    if k == 0:
        threshold = np.inf
    elif k == len(all_coeffs):
        threshold = -1.0
    else:
        # Quick select or sort
        # For speed, np.partition
        partitioned = np.partition(abs_coeffs, -k)
        threshold = partitioned[-k]
        
    # Masking
    mask_real = np.abs(dct_real) >= threshold
    mask_imag = np.abs(dct_imag) >= threshold
    
    # Keep coefficients
    dct_real_kept = dct_real * mask_real
    dct_imag_kept = dct_imag * mask_imag
    
    # 4. Quantize and Encode
    # We need to encode:
    # 1. The Mask (Bitmap or Run-Length)
    # 2. The Values (Quantized)
    
    # For simulation purposes:
    # Bits = (Number of Non-Zero Coeffs) * (Bits per Coeff) + Overhead for Mask
    # Overhead for mask: simple bitmap is 1 bit per pixel (H*W*N*2). 
    # Compressed bitmap is usually much smaller for sparse data.
    # Let's assume 1 bit per coeff overhead as a conservative/simple estimate for location info,
    # or just use the zlib compression of the sparse array which handles zeros efficiently?
    # Our quantize_and_encode function currently compresses the whole array (including zeros).
    # If zeros are frequent, zlib will compress them well.
    # So we can just pass the sparse arrays to `quantize_and_encode`.
    
    # Combine into complex for quantizer (it handles real/imag separation)
    coeffs_kept = dct_real_kept + 1j * dct_imag_kept
    
    # Note: quantize_and_encode separates real/imag and zlib compresses them.
    # zlib on an array with many zeros is effectively RLE.
    rec_coeffs, bits = quantize_and_encode(coeffs_kept, bits=quant_bits)
    
    # 5. Inverse DCT
    rec_real = idctn(rec_coeffs.real, axes=(-2, -1), norm='ortho')
    rec_imag = idctn(rec_coeffs.imag, axes=(-2, -1), norm='ortho')
    
    rec_imgs = rec_real + 1j * rec_imag
    
    return rec_imgs, bits

def run_dct_experiment():
    imgs = load_imgs()
    ref_img_path = os.path.join("results", "refer0", "ref_image.pt")
    try:
        ref_img = torch.load(ref_img_path).numpy()
    except:
        print(f"Reference image not found at {ref_img_path}")
        return
         
    N, H, W = imgs.shape
    num_pixels = H * W
    num_pixels_total = N * num_pixels
    
    # Sweep keep ratios
    ratios = [0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    quant_bits = 9
    
    results = {'bpp': [], 'psnr': [], 'ssim': [], 'ratio': []}
    
    # Create output directory
    output_dir = os.path.join("results", "refer1b_dct_compression")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting DCT (Water-filling) compression sweep...")
    
    for r in ratios:
        print(f"\n--- Keep Ratio: {r} ---")
        
        rec_imgs, bits = dct_compress_decompress(imgs, r, quant_bits)
        bpp = bits / num_pixels_total
        
        p, s = run_espirit_pipeline(rec_imgs, ref_img, verbose=False)
        
        results['bpp'].append(bpp)
        results['psnr'].append(p)
        results['ssim'].append(s)
        results['ratio'].append(r)
        
        print(f"BPP: {bpp:.2f}, PSNR: {p:.2f} dB, SSIM: {s:.4f}")
        
        # Reconstruct and Plot for this ratio
        # We need to run the ESPIRiT combination step to get the final image
        from project_JH.utils.espirit_torch import csm_from_espirit
        from project_JH.utils.mri_utils import crop_center, get_device
        
        # 1. FFT
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
        
        # Save reconstruction
        rec_path = os.path.join(output_dir, f"rec_img_r{r}.pt")
        torch.save(torch.from_numpy(rec_final), rec_path)
        
        # Plot reconstruction
        plt.figure(figsize=(5, 5))
        plt.imshow(np.abs(rec_final), cmap='gray')
        plt.title(f"DCT Ratio={r}\nBPP={bpp:.2f}, PSNR={p:.2f}dB, SSIM={s:.4f}")
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f"rec_img_r{r}.png"), bbox_inches='tight')
        plt.close()
        
    torch.save(results, os.path.join(output_dir, "results_dct.pt"))
    save_rd_curve(results, "DCT", "dct_rd_curve.png", output_dir=output_dir)
    print("Done.")

if __name__ == "__main__":
    run_dct_experiment()

