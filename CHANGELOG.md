# Changelog

## Unreleased

- Choose memory use automatically (`tools/idol_memory.py`): the server measures the free graphics memory with `nvidia-smi` (or PyTorch) and keeps every model resident, with batched guidance and cuDNN attention, when it fits, otherwise stages them; the image generator's budget is sized for each image from the free memory, and the Idol session is kept loaded during a prompt when both fit. New `--memory-mode`, `--[no-]batch-guidance`, `--[no-]cudnn-attention` and `--vram-budget-gib` options; `max_vram_gib` and `setup_iris.py --max-vram-gib` accept `auto`, now the default. `/health` reports `memory_mode`. Measured peaks are in `docs/data/memory-tiers.json`.
- Measure the published `Idol-Q8_0.gguf` against a build without the denoiser's covariance calibration on 36 benchmark cases: the build without it is not better, so the published file is unchanged (`docs/data/q8-denoiser-calibration-ablation.json`).
- Measure the staged low-memory path with the published file (`docs/data/idol-staged-memory.json`) and compare the native CPU decoder with the GPU decoders (`docs/data/decoder-cpu-vs-gpu.json`).
- Ignore local build variants, the cloned upstream SDK, and test caches.
- Build the Idol pipeline on the CPU and run the image encoders one at a time when inference is staged (`tools/idol_staging.py`, `tools/idol_generate_staged.py`, and the server's default mode), lowering peak graphics memory; add tests and a measurement record.
- Add `--max-vram-gib` to `tools/setup_iris.py` and document how the image generator's graphics-memory budget affects peak memory and speed, with a measurement record.
- Install `python3-dev` in the Docker image so Triton can compile its CUDA helper and the fused-kernel path runs in the container.
- Build a pinned GGML (v0.29.0) shared library in the Docker image so the K-block quantization tests run there instead of skipping.

## 0.1.0 (2026-11-06)

- Added a 32-object, 96-case matched-seed fidelity benchmark, including mesh and approximate-view comparisons and the observed worst case.
- Applied the Apache License 2.0 to the project source and retained separate third-party notices.
- Added sampled end-to-end prompt memory measurements and a bounded Docker reconstruction and mesh test record.
- Added a browser Mesh tab with native extraction, saved Gaussian uploads, mesh downloads, reference previews, and downloadable comparisons.
- Added sampled point-to-triangle geometry distances and threshold F1 scores, alongside the older point-sampling diagnostics, with measured original-versus-Q8 examples.
- Added full Idol model export to Georgi Gerganov's Unified Format (GGUF), preserving the original model configuration, component names, and source hashes.
- Added an image-to-Gaussian generation path using the upstream Idol software development kit and the native octree and Gaussian decoders.
- Added a complete calibrated Q8 model covering the denoiser, background remover, octree, and Gaussian decoder.
- Added construction comparisons and tests for the model container, decoder conversion, and guarded inference hook.
- Added a local server, browser interface, and configuration for a separate Iris image generator.
