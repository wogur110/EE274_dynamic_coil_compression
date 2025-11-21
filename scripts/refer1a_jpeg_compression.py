import torch
import numpy as np
import sigpy as sp
import os
import sys
import io
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project_JH to path (go up 3 levels from scripts/ to project_JH/)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_jh_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
if project_jh_dir not in sys.path:
    sys.path.insert(0, project_jh_dir)

from project_JH.utils.mri_utils import (
    load_imgs,
    get_device,
    run_espirit_pipeline,
    DATA_DIR,
    normalize_complex_image,
)
from project_JH.utils.plot_utils import save_rd_curve

# Moving tile/detile to this file or assume they are in common if I added them. 
# I didn't add them to common yet. Let's add them to common first or just keep here.
# Since JPEG is the only one tiling 8x8, maybe keep here. But "coil-agnostic JPEG baseline" might be reused?
# I'll keep them here for now or define them locally.

def tile_images(imgs):
    """
    Tiles a stack of images (N, H, W) into a mosaic (H_mosaic, W_mosaic).
    """
    N, H, W = imgs.shape
    n_side = int(np.ceil(np.sqrt(N)))
    
    mosaic_H = n_side * H
    mosaic_W = n_side * W
    
    mosaic = np.zeros((mosaic_H, mosaic_W), dtype=imgs.dtype)
    
    for i in range(N):
        r = i // n_side
        c = i % n_side
        mosaic[r*H:(r+1)*H, c*W:(c+1)*W] = imgs[i]
        
    return mosaic

def detile_images(mosaic, N, H, W):
    """
    Detiles a mosaic back into a stack of images (N, H, W).
    """
    n_side = int(np.ceil(np.sqrt(N)))
    imgs = np.zeros((N, H, W), dtype=mosaic.dtype)
    
    for i in range(N):
        r = i // n_side
        c = i % n_side
        imgs[i] = mosaic[r*H:(r+1)*H, c*W:(c+1)*W]
        
    return imgs

def quantize_to_uint8(x, min_val, max_val):
    x = np.clip(x, min_val, max_val)
    x_norm = (x - min_val) / (max_val - min_val)
    return (x_norm * 255).astype(np.uint8)

def dequantize_from_uint8(x_uint8, min_val, max_val):
    x_norm = x_uint8.astype(np.float32) / 255.0
    return x_norm * (max_val - min_val) + min_val

def jpeg_compress_decompress(mosaic, quality):
    min_val = mosaic.min()
    max_val = mosaic.max()
    
    mosaic_uint8 = quantize_to_uint8(mosaic, min_val, max_val)
    img_pil = Image.fromarray(mosaic_uint8, mode='L')
    
    buffer = io.BytesIO()
    img_pil.save(buffer, format='JPEG', quality=quality)
    size_bytes = buffer.tell()
    
    buffer.seek(0)
    img_rec_pil = Image.open(buffer)
    mosaic_rec_uint8 = np.array(img_rec_pil)
    
    mosaic_rec = dequantize_from_uint8(mosaic_rec_uint8, min_val, max_val)
    return mosaic_rec, size_bytes * 8

def run_jpeg_experiment():
    # Load data
    imgs = load_imgs()
    ref_img_path = os.path.join("results", "refer0", "ref_image.pt")
    
    print(f"Loading reference from {ref_img_path}...")
    ref_img = torch.load(ref_img_path)
    if isinstance(ref_img, torch.Tensor):
        ref_img = ref_img.numpy()
        
    N, H, W = imgs.shape
    num_pixels = H * W
    num_pixels_total = N * num_pixels
    
    # Expanded quality range
    qualities = [1, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95, 98, 100]
    
    results = {'bpp': [], 'psnr': [], 'ssim': [], 'quality': []}
    
    imgs_real = imgs.real
    imgs_imag = imgs.imag
    
    mosaic_real = tile_images(imgs_real)
    mosaic_imag = tile_images(imgs_imag)
    
    # Create output directory
    output_dir = os.path.join("results", "refer1a_jpeg_compression")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting JPEG compression sweep...")
    
    for q in tqdm(qualities):
        print(f"\n--- Quality: {q} ---")
        
        rec_mosaic_real, bits_real = jpeg_compress_decompress(mosaic_real, q)
        rec_mosaic_imag, bits_imag = jpeg_compress_decompress(mosaic_imag, q)
        
        total_bits = bits_real + bits_imag
        bpp = total_bits / num_pixels_total  # Bits per complex coil pixel
        
        rec_imgs_real = detile_images(rec_mosaic_real, N, H, W)
        rec_imgs_imag = detile_images(rec_mosaic_imag, N, H, W)
        
        rec_imgs = rec_imgs_real + 1j * rec_imgs_imag
        
        # Run ESPIRiT to get final image
        # We need the reconstructed image to plot it
        # run_espirit_pipeline calculates metrics but we might want the image back?
        # The current run_espirit_pipeline returns only metrics.
        # Let's call it to get metrics first.
        p, s = run_espirit_pipeline(rec_imgs, ref_img, verbose=False)
        
        results['bpp'].append(bpp)
        results['psnr'].append(p)
        results['ssim'].append(s)
        results['quality'].append(q)
        
        print(f"BPP: {bpp:.2f}, PSNR: {p:.2f} dB, SSIM: {s:.4f}")
        
        # To plot reconstruction, we need to run ESPIRiT and get the image.
        # Since run_espirit_pipeline does this internally, we can either duplicate code 
        # or modify run_espirit_pipeline. For now, let's just use the same logic locally 
        # or accept that we re-run it if we want the image.
        # Actually, let's just re-run the minimal reconstruction part here to get the image for plotting.
        # Wait, run_espirit_pipeline is expensive.
        # Let's modify run_espirit_pipeline or just copy the recon logic here.
        # Copying logic is safer to avoid breaking other scripts.
        
        # Reconstruct for plotting/saving
        from project_JH.utils.espirit_torch import csm_from_espirit
        from project_JH.utils.mri_utils import crop_center
        
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
        rec_path = os.path.join(output_dir, f"rec_img_q{q}.pt")
        torch.save(torch.from_numpy(rec_final), rec_path)
        
        # Plot reconstruction
        plt.figure(figsize=(5, 5))
        plt.imshow(np.abs(rec_final), cmap='gray')
        plt.title(f"JPEG Q={q}\nBPP={bpp:.2f}, PSNR={p:.2f}dB, SSIM={s:.4f}")
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f"rec_img_q{q}.png"), bbox_inches='tight')
        plt.close()

    # Save results
    torch.save(results, os.path.join(output_dir, "results_jpeg.pt"))
    
    # Plot RD Curve
    save_rd_curve(results, "JPEG", "jpeg_rd_curve.png", output_dir=output_dir)
    print("Done.")

if __name__ == "__main__":
    run_jpeg_experiment()
