# Changelog

## 0.1.0

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
