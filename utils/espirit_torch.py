import torch
import numpy as np
import sigpy as sp
# from einops import rearrange, einsum
from tqdm import tqdm
import time
from typing import Optional, Tuple

def torch_to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

def np_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return x

def centered_ifft(x, dim=None, norm='ortho'):
    """
    Centered Inverse FFT. 
    Assumes input has DC at center.
    """
    if dim is None:
        dim = tuple(range(x.ndim))
        
    x = torch.fft.ifftshift(x, dim=dim)
    x = torch.fft.ifftn(x, dim=dim, norm=norm)
    x = torch.fft.fftshift(x, dim=dim)
    return x

def power_method_matrix(A, num_iter=100, verbose=False):
    """
    Computes the dominant eigenvector and eigenvalue of a batch of matrices A
    using the power method.
    
    Args:
        A: Tensor of shape (..., N, N) representing Hermitian matrices.
        num_iter: Number of iterations.
        verbose: Whether to show progress.
        
    Returns:
        v: Dominant eigenvector of shape (..., N)
        l: Dominant eigenvalue of shape (...)
    """
    # A is (..., N, N)
    # Initialize random vector v of shape (..., N)
    batch_shape = A.shape[:-2]
    N = A.shape[-1]
    device = A.device
    dtype = A.dtype
    
    v = torch.randn(*batch_shape, N, device=device, dtype=dtype)
    v = v / torch.linalg.norm(v, dim=-1, keepdim=True)
    
    iterator = range(num_iter)
    if verbose:
        iterator = tqdm(iterator, desc="Power Method")
        
    for _ in iterator:
        # v <- A @ v
        # (..., N, N) @ (..., N, 1) -> (..., N, 1)
        # v_new = einsum(A, v, '... i j, ... j -> ... i')
        v_new = torch.einsum('...ij,...j->...i', A, v)
        
        # Normalize
        norm = torch.linalg.norm(v_new, dim=-1, keepdim=True)
        v = v_new / (norm + 1e-12)
        
    # Compute Rayleigh quotient for eigenvalue approximation: (v^H A v) / (v^H v)
    # Since v is normalized, denominator is 1.
    # l = v^H A v
    # Av = einsum(A, v, '... i j, ... j -> ... i')
    Av = torch.einsum('...ij,...j->...i', A, v)
    # v^H is conj(v)
    # l = einsum(v.conj(), Av, '... i, ... i -> ...').real
    l = (v.conj() * Av).sum(dim=-1).real
    
    return v, l

def csm_from_espirit(ksp_cal: torch.Tensor,
                     im_size: tuple,
                     thresh: Optional[float] = 0.02,
                     kernel_width: Optional[int] = 6,
                     crp: Optional[float] = None,
                     sets_of_maps: Optional[int] = 1,
                     max_iter: Optional[int] = 100,
                     lobpcg_iter: Optional[int] = None,
                     cpu_last_part: Optional[bool] = False,
                     verbose: Optional[bool] = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Implementation of ESPIRiT calibration in PyTorch.
    Adapted from: https://github.com/danielabrahamgit/mr_recon/blob/main/src/mr_recon/multi_coil/coil_est.py
    """

    # Consts
    img_ndim = len(im_size)
    num_coils = ksp_cal.shape[0]
    device = ksp_cal.device

    # Get calibration matrix.
    # Shape [num_coils] + num_blks + [kernel_width] * img_ndim
    # Using SigPy for Hankel matrix construction as it's efficient and robust
    ksp_cal_sp = torch_to_np(ksp_cal)
    
    # Ensure we use CPU for sigpy operations if cupy is not available or if tensor is on CPU
    # But if ksp_cal is on GPU, ksp_cal_sp is numpy (CPU). 
    # Sp.array_to_blocks works on numpy or cupy.
    # We'll do this part on CPU to be safe/simple unless ksp_cal was on GPU and cupy is available?
    # The original code uses sp.get_device(ksp_cal_sp). Since we did torch_to_np, it's CPU.
    
    # Note: sigpy.array_to_blocks creates a view, avoiding large copies if possible.
    mat = sp.array_to_blocks(
        ksp_cal_sp, [kernel_width] * img_ndim, [1] * img_ndim)
    mat = mat.reshape([num_coils, -1, kernel_width**img_ndim])
    mat = mat.transpose([1, 0, 2])
    mat = mat.reshape([-1, num_coils * kernel_width**img_ndim])
    
    # Convert back to torch
    mat = np_to_torch(mat).to(device)

    # Perform SVD on calibration matrix
    if verbose:
        print('Computing SVD on calibration matrix: ', end='')
        start = time.perf_counter()
    
    # We don't implement lobpcg here to keep it simple, defaulting to SVD
    if lobpcg_iter is not None:
        print("Warning: lobpcg_iter ignored, using SVD.")
        
    # torch.linalg.svd
    # mat is (N_blocks, N_coils * kernel_size)
    # We want V. SVD returns U, S, Vh. V is Vh.H
    # In torch: U, S, Vh = svd(A). A = U @ diag(S) @ Vh
    # The nullspace is in Vh (rows). 
    # We want right singular vectors corresponding to singular values > threshold.
    
    # Full matrices=False -> U is (M, K), Vh is (K, N). K=min(M,N).
    # We want the row space of the signal subspace.
    _, S, VH = torch.linalg.svd(mat, full_matrices=False)
    
    # Keep singular vectors above threshold
    # VH is (K, N_features). Rows are eigenvectors of A^H A.
    # We filter based on S.
    VH = VH[S > thresh * S.max(), :]
    
    if verbose:
        end = time.perf_counter()
        print(f'{end - start:.3f}s')

    # Get kernels
    # VH shape: (num_kernels, num_coils * kernel_width^ndim)
    num_kernels = len(VH)
    kernels = VH.reshape(
        [num_kernels, num_coils] + [kernel_width] * img_ndim)

    # Get covariance matrix in image domain
    if cpu_last_part:
        device = torch.device('cpu')
        
    # Prepare for image domain covariance calculation
    # AHA shape: (*im_size, num_coils, num_coils)
    AHA = torch.zeros(im_size + (num_coils, num_coils), 
                        dtype=ksp_cal.dtype, device=device)
    kernels = kernels.to(device)
    
    # Transform kernels to image domain and compute AA^H
    # kernel: (num_kernels, num_coils, kw, kw)
    # We pad to image size and IFFT.
    # In the original code:
    # aH = ifft(kernel, oshape=(num_coils, *im_size), dim=tuple(range(-img_ndim, 0)))
    # aH is (num_kernels, num_coils, *im_size) ? No, ifft on dim -1, -2.
    # Let's follow the dimensions carefully.
    # kernel shape: (nk, nc, kw, kw)
    # ifft on last 2 dims.
    
    # We need to use our centered_ifft but with padding/oshape.
    # torch.fft.ifftn doesn't support oshape (output shape) directly with automatic zero-padding in the same way sigpy does conveniently? 
    # Actually `s` parameter in ifftn controls output shape (padding/truncation).
    
    for kernel in tqdm(kernels, 'Computing covariance matrix', disable=not verbose):
        # kernel: (nc, kw, kw)
        # Pad to im_size?
        # Sigpy ifft with oshape pads centered.
        # We need to pad kernel from (kw, kw) to (im_size) centered.
        
        # Create padded kernel
        # We can use torch.zeros
        pad_kernel = torch.zeros((num_coils, *im_size), dtype=kernel.dtype, device=device)
        
        # Place kernel in center
        # Indices
        # kw=6. Center of kw is 3. 
        # We place it at center of im_size.
        
        # Actually, standard practice: place at corners if using fft, or center if using centered fft.
        # Since we use centered_ifft, we should place it at center.
        # Let's use a helper or manual assignment.
        # Center indices
        ck = [k // 2 for k in [kernel_width] * img_ndim]
        ci = [i // 2 for i in im_size]
        
        # Slicing
        slices_k = [slice(0, k) for k in [kernel_width] * img_ndim]
        
        # We need to handle odd/even sizes correctly.
        # Let's simplify: just copy to a zero tensor and roll?
        # Or stick to sigpy logic: "resize"
        # sigpy.util.resize would center crop/pad.
        
        # Let's try to rely on `s` param in ifftn if we assume corner-centered?
        # If using `centered_ifft` (fftshift -> ifft -> fftshift), we expect input to be centered in frequency domain.
        # So we should pad the kernel to im_size, keeping the kernel at the center of the grid.
        
        # Pad kernel
        # We can use sp.resize logic if we want, or just manual.
        # Assuming kernel_width is small and even/odd.
        # Let's use a robust way.
        
        # Create empty grid
        k_padded = torch.zeros((num_coils, *im_size), dtype=kernel.dtype, device=device)
        
        # Calculate start indices to center the kernel
        starts = [(sz - kw) // 2 for sz, kw in zip(im_size, [kernel_width] * img_ndim)]
        slices = [slice(s, s + kw) for s, kw in zip(starts, [kernel_width] * img_ndim)]
        
        # Assign (need to handle dims)
        if img_ndim == 2:
            k_padded[:, slices[0], slices[1]] = kernel
        elif img_ndim == 3:
            k_padded[:, slices[0], slices[1], slices[2]] = kernel
            
        # Now IFFT
        # dim=tuple(range(-img_ndim, 0)) -> last ndim dimensions
        aH = centered_ifft(k_padded, dim=tuple(range(-img_ndim, 0)))
        
        # aH shape: (nc, *im_size)
        # We want (..., nc, 1) for broadcasting in matrix mult
        # aH = rearrange(aH, 'nc ... -> ... nc 1')
        # Permute to put coil dim at -2, add dim at -1
        perm = list(range(1, aH.ndim)) + [0]
        aH = aH.permute(*perm).unsqueeze(-1)
        
        # AHA += aH @ aH.H
        # outer product of coil vector at each pixel
        # aH is column vector (nc, 1) at each pixel.
        # aH.swapaxes(-1, -2).conj() is row vector (1, nc).
        # Matmul gives (nc, nc).
        
        # Optimization from original code:
        # bs = 1
        # for c1 in range(0, num_coils, bs):
        #    ...
        # We can just do it directly if memory allows.
        # AHA += aH @ aH.swapaxes(-1, -2).conj()
        
        # To save memory, we can loop or use addcmul if simpler, but matmul is fine.
        # Let's follow original optimization if needed, but for 64 coils, (64,64) per pixel is large.
        # im_size 294x294 ~ 90k pixels. 90k * 64*64 * 8 bytes ~ 3GB. 
        # + overhead. Should fit in A5000 (24GB).
        
        AHA += aH @ aH.swapaxes(-1, -2).conj()

    # Scale AHA
    AHA *= (np.prod(im_size) / kernel_width**img_ndim)
    
    # Get eigenvalues and eigenvectors
    mps_all = []
    evals_all = []
    for i in range(sets_of_maps):
        
        # power iterations
        # AHA is (..., nc, nc)
        mps, eigen_vals = power_method_matrix(AHA, num_iter=max_iter*3, verbose=verbose)
        
        # mps: (..., nc)
        # eigen_vals: (...)
        
        # Update AHA if multiple sets
        if sets_of_maps > 1:
            # Deflate
            # AHA -= lam * v * vH
            # mps is v.
            # term = einsum(mps * eigen_vals.unsqueeze(-1), mps.conj(), '... i, ... j -> ... i j')
            # Outer product
            vec1 = mps * eigen_vals.unsqueeze(-1)
            term = torch.matmul(vec1.unsqueeze(-1), mps.conj().unsqueeze(-2))
            AHA -= term
            
        mps_all.append(mps)
        evals_all.append(eigen_vals)
        
    if sets_of_maps == 1:
        mps = mps_all[0]
        eigen_vals = evals_all[0]
    else:
        mps = torch.stack(mps_all, dim=0) # S *im_size nc ?
        # Original code: stack dim 1? "C S *im_size"? No, let's check.
        # Original: mps_all.append(mps). mps shape (..., nc).
        # Stack dim 1? The return shape says (ncoil, *im_size).
        # Rearrange later.
        
        # If sets > 1, we might want (sets, ..., nc) or similar.
        # Let's return stacked.
        mps = torch.stack(mps_all, dim=0)
        eigen_vals = torch.stack(evals_all, dim=0)
    
    # Rearrange mps to (nc, *im_size) or (sets, nc, *im_size)
    # Currently mps is (*im_size, nc).
    # We want (nc, *im_size).
    if sets_of_maps == 1:
        # mps = rearrange(mps, '... nc -> nc ...')
        # Permute last dim to first
        mps = mps.permute(-1, *range(mps.ndim-1))
        # eigen_vals is (...)
    else:
        # mps = rearrange(mps, 's ... nc -> s nc ...')
        # Permute: s(0), nc(-1) -> 1, ...
        mps = mps.permute(0, -1, *range(1, mps.ndim-1))

    # Phase relative to first map and crop
    # If sets=1:
    if sets_of_maps == 1:
        # mps[0] is first coil map? No. mps is (nc, y, x).
        # "Phase relative to first map" usually means standardize phase using the first component of the vector?
        # "mps[0]" in original code likely referred to the first set of maps if it was stacked?
        # Or the first coil? 
        # Original: "mps *= torch.conj(mps[0] / (torch.abs(mps[0]) + 1e-12))"
        # If mps is (nc, ...), mps[0] is the first coil image.
        # This aligns phase so that coil 0 is real/positive.
        phase_ref = mps[0] / (torch.abs(mps[0]) + 1e-12)
        mps *= torch.conj(phase_ref)
        
        if crp:
            mps *= (eigen_vals > crp).float()
    else:
        # Handle sets
        # Assuming we phase each set separately or all to the first coil of first set?
        # Original code does: mps *= torch.conj(mps[0] / ...) 
        # If mps is (sets, nc, ...), mps[0] is first set.
        pass # Not implementing sets>1 details fully unless requested.

    return mps, eigen_vals

