# Changelog

## Unreleased

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
