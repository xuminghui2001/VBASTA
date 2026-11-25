# VBASTA

Official MATLAB implementation of the paper:

> **Structured Sparse Representation Learning with Locality-Preserving Strict Sparsity and Spatially Adaptive Geometry**  
> Jing Li\*, Minghui Xu, Jinjia Wang  
> *Pattern Recognition*, 2026  
> DOI: [10.1016/j.patcog.2026.114878](https://doi.org/10.1016/j.patcog.2026.114878)

## Repository Structure

All source codes and test datasets are located in `V-BASTA局部处理l0/`:

- `main_VBASTA_L0_denoising.m`: Main demo script for image denoising experiments.
- `main_inpainting_VBASTA_L0_dy.m`: Main demo script for image inpainting tasks.
- `main_VBASTA_L0_dy2.m`: Dictionary learning and sparse representation experiments.
- `main_VBASTA_lcn_inverse.m`: Image reconstruction with local contrast normalization.
- `prox_vmetric_L2_ball.m`, `proxl0.m`, `compute_H_gradients_in_parallel.m`: Core variable-metric proximal operators and parallel gradient routines.
- `Util/` & `image_helpers/`: Auxiliary image processing, contrast normalization, and evaluation utilities (PSNR, MSE, SNR).
- `mexfiles/`: C/MEX acceleration routines (`im2colstep`, `col2imstep`, etc.) with precompiled `.mexw64` binaries and `make.m`.
- `vlfeat/`: Bundled VLFeat computer vision library.
- `datasets/`: Benchmark test images (`CSBD68`, `standard`, `separation`, `city_100_100`, `fruit_100_100`).

## Requirements & Quick Start

1. **Environment**: MATLAB (tested on R2020b and later).
2. **MEX Compilation**: Precompiled 64-bit Windows binaries (`.mexw64`) are already included in `mexfiles/`. If you are running on Linux or macOS, navigate to `mexfiles/` in MATLAB and run:
   ```matlab
   make
