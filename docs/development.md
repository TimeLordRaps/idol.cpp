# Evaluation and quantization development

Run the commands below from the repository root. These experiments are not release recommendations unless stated otherwise.

## Mesh extraction and fidelity

The browser Mesh tab uses the native `idol_mesh` executable. Supply `--mesh-executable /path/to/idol_mesh` if it is not in the build directories or executable search path. Extraction runs on the processor with two OpenMP threads and a 120-second timeout; the interface permits grid resolutions from 32 to 256 cells per axis. The source Gaussian file is preserved. Each mesh has a `.mesh.json` sidecar recording source and output digests, native-executable digest, settings, elapsed time, and geometry counts.

For a saved Gaussian cloud:

```bash
curl -F splats=@result.ply -F resolution=176 \
  http://127.0.0.1:8127/v1/meshes -o result.obj
```

For an existing server result, send JSON such as `{"resolution":176,"sigma":1.5,"iso":0.2,"min_piece":0.05}` to `POST /v1/reconstructions/{result_id}/mesh`. The response contains the mesh and an `X-Idol-Mesh-Metadata` header identifying `/v1/meshes/{mesh_id}/metadata`. Submit a multipart `reference` object file to `/v1/meshes/{mesh_id}/comparisons`; optional `samples` and `seed` fields default to 20,000 and 23. Mesh uploads are limited to 32 MiB and saved Gaussian uploads to 80 MiB.

The downloadable report includes both the primary `sampled_point_to_triangle` metrics and legacy sampled-point-to-sampled-point diagnostics. The latter include target-sampling noise; use the triangle-distance fields when reporting geometric fidelity. The [trimesh proximity query](https://trimesh.org/trimesh.proximity.html) finds the closest target triangles, accelerated by the separately installed `rtree` package. Source points remain finitely sampled by triangle area. Distances are unsquared Euclidean distances in model units; the symmetric mean averages the two directional means. The symmetric 95th percentile is the larger directional percentile. The sampled maximum does not establish the maximum over the entire continuous surface.

Match input image, model-coordinate scale, preprocessing mode, seed, steps, Gaussian count, runtime settings, and every extraction option when assessing quantization. Mesh filtering and smoothing can suppress differences; compare appearance separately. The [Q8 mesh report](data/q8-mesh-fidelity.json) records two development cases, not a held-out ground-truth benchmark.

## Local server and browser interface

Install the optional browser dependencies with `python -m pip install -r requirements-server.txt`, then start the server:

```bash
sh idol-server --model /path/to/Idol-model.gguf
```

On Windows use `.\idol-server.cmd --model C:\path\to\Idol-model.gguf`; running `python tools/idol_server.py` directly is also supported. Open `http://127.0.0.1:8127` to upload an image or capture one from a webcam, reconstruct it, and inspect the resulting 3D model. Set `--port` to choose another port, `--host` to change the listening address, and `--output-dir` to choose where completed point clouds are saved. The server loads the model on the first request and reuses it for later requests; add `--preload` to load it before accepting requests. It serves one reconstruction at a time; actual latency and memory use depend on the model and hardware. The original-precision image encoders and their separate access requirements still apply.

`--repo TimeLordRaps/Idol-GGUF --file Idol-Q8_0.gguf` downloads the released model from Hugging Face instead of using `--model`; `--revision` selects a repository revision.

`--memory-mode` chooses how Idol uses graphics memory. The default, `auto`, measures the free memory (`tools/idol_memory.py`, via `nvidia-smi` or PyTorch) and uses `resident` when at least 6,488 MiB are free and `staged` otherwise; it also turns batched guidance and cuDNN attention on in resident mode. `resident` (alias `--resident-gpu`) keeps the image pipeline and decoders on the graphics device between phases, reducing host transfers at the cost of higher graphics-memory use; an explicit mode leaves the speed options off unless they are requested. `staged` builds every model on the CPU and runs one stage at a time, keeping the models that are waiting in memory. `lean` does the same but keeps them in temporary files (`tools/idol_lowram.py`; `--lean-directory` chooses where, about 3.7 GB), lowering host memory from 5,183 to 3,286 MiB at the cost of reading them back for every generation; `auto` picks it when the computer has less than 6,207 MiB free and enough disk space, and it does not support `--packed-denoiser` or `--fused-kernels`. `--vram-budget-gib` states how much memory may be counted on instead of the measured free amount. The thresholds are the whole-process peaks in [memory-tiers.json](data/memory-tiers.json) plus a 512 MiB headroom; accuracy, speed, and peak memory still need measurement for other hardware and models.

The experimental `--batch-guidance` option combines the conditional and unconditional denoising passes into a batch. It can change floating-point results and peak graphics-memory use; measure fidelity and speed for the chosen hardware and model.

The experimental `--cudnn-attention` option also selects attention from NVIDIA's CUDA Deep Neural Network library (cuDNN) and TensorFloat-32 (TF32) matrix multiplication during sampling, using the upstream optimized sampling settings without compiling the model. It implies batched guidance and requires a compatible CUDA and cuDNN installation; measure its fidelity and speed before use.

Idol itself takes images rather than text, and no image-generation model is bundled. To enable prompts, configure a local [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) command-line runner with `--image-config /path/to/image-generator.json`, or supply a compatible service with `--image-server http://127.0.0.1:PORT`. The default configuration is `%LOCALAPPDATA%\idol.cpp\image-generator.json` on Windows and `${XDG_CONFIG_HOME:-~/.config}/idol.cpp/image-generator.json` elsewhere. Local configuration specifies the runner, diffusion model, text encoder, and an image decoder when the selected model needs one. Iris-3B generates pixels directly and must not be given a separate decoder. The local runner exits before Idol loads; an existing Idol session is released before the next prompt to avoid keeping both models on the graphics device. This reduces overlap but adds loading time.

The service adapter uses `/v1/images/generations` and requires `generation_info.models` to identify the image-generation model. The browser shows the generated input image and the configured local filename or service-reported filename. Result metadata records component filenames, the original image digest, and the digest of the displayed PNG, which may have been re-encoded. Filenames are identifiers, not independently verified weight hashes. Each prompt produces a new image and reconstruction; it does not edit an existing 3D model. Image-model licenses and hardware requirements are separate from Idol's.

For the tested [Iris-3B Q4](https://huggingface.co/webmp3/Sakura-Iris-3B-GGUF) and [Qwen3-VL 4B Q4 text encoder](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF), both distributed under Apache 2.0, install an Iris-compatible `sd-cli`, then run:

```bash
python tools/setup_iris.py --runner /path/to/sd-cli
```

The setup downloads the two pinned model files into the Hugging Face cache, checks their Secure Hash Algorithm 256-bit (SHA-256) digests, and writes the default configuration without overwriting an existing one. The files total about 5.9 GB. Use `--image-model` and `--text-encoder` to reuse the exact tested files already on disk, or `--config` to choose another configuration path. The runner and model downloads are separate from Idol's gated image encoders. The runner and models were exercised with stable-diffusion.cpp release `master-1052-f5df5f8`; compatibility with other versions is not established.

An equivalent manual configuration uses absolute paths:

```json
{
  "version": 1,
  "engine": "stable-diffusion.cpp",
  "executable": "/path/to/sd-cli",
  "diffusion_model": "/path/to/Sakura-Iris-3B-Q4_K_M-3.14GiB.gguf",
  "text_encoder": "/path/to/Qwen3VL-4B-Instruct-Q4_K_M.gguf",
  "license_notice": "Iris-3B and Qwen3-VL-4B: Apache-2.0",
  "steps": 30,
  "cfg": 3.0,
  "sampling_method": "dpm++2m",
  "width": 1024,
  "height": 1024,
  "max_vram_gib": 16
}
```

The `cfg` value controls classifier-free guidance, which adjusts how strongly generation follows the prompt. `max_vram_gib` is the runner's graphics-memory budget in gibibytes (one gibibyte is 2³⁰ bytes), or `"auto"` (the default when it is absent), which sizes the budget for every image from the free graphics memory: 16 when 7,222 MiB or more are free, otherwise the largest of 6, 5, 4, 3 or 2 whose measured peak fits (2 if none does). It is a budget for managed weights and runner buffers, not a hard limit. Lower values make the runner load weights on demand and run the graph in segments, which reduces peak memory and slows generation; see the [measurements](data/iris-memory-budget.json) for 16, 3, 2 and 1 and [memory-tiers.json](data/memory-tiers.json) for 16, 8, 6, 5 and 4. `mmap` is `"auto"` (the default when it is absent), `true` or `false`: whether the runner memory-maps its weights (`--mmap`), so that weights it keeps off the graphics device are file pages the system can drop and read again instead of private memory. `"auto"` does it only when the free system memory is less than the private memory the runner held at the graphics budget in use (6,737 to 6,898 MiB at 2 to 4 GiB, 3,547 MiB at 6 GiB, 1,093 MiB at 16 GiB) plus a 1,024 MiB headroom; `setup_iris.py --mmap auto|on|off` writes it. See the [text-prompt measurements](data/text-prompt-host-memory.json). Other image models can require a separate `vae` file and different sampling settings.

The application programming interface (API) is available on the same port. `GET /health` reports whether the model is loaded. Send a multipart image to `POST /v1/reconstructions`, or JSON containing `prompt` to `POST /v1/text-to-3d` when an image generator is configured. Both return a PLY file and an `X-Idol-Metadata` header pointing to the result metadata; prompt results also include the URL-encoded `X-Idol-Image-Model` and generated-image `X-Idol-Input` headers. Optional fields are `mode`, `seed`, `steps`, and `gaussians`. The service has no authentication: its default loopback address keeps it local; exposing it on a network requires an authenticated proxy.

```bash
curl -F image=@photo.jpg http://127.0.0.1:8127/v1/reconstructions -o result.ply
curl -H 'Content-Type: application/json' -d '{"prompt":"a red chair"}' \
  http://127.0.0.1:8127/v1/text-to-3d -o prompt-result.ply
```

## Reusable Python session

For sequential requests on a compatible NVIDIA GPU, keep the image pipeline and decoders loaded in a reusable session. It stages models through host memory between inference phases and writes one new PLY file per call:

```python
import sys
sys.path.insert(0, "tools")
from idol_session import IdolSession

with IdolSession("/path/to/Idol-model.gguf") as session:
    first = session.generate("photo-1.jpg", "result-1.ply", mode="real")
    second = session.generate("photo-2.jpg", "result-2.ply", mode="real")
```

Each result contains `latent`, `preprocessed_image`, and `camera`. Existing output files are rejected. A failed inference invalidates the session; create a new one before retrying. Run inference sequentially within the process because sampling uses PyTorch's process-wide random generator. The original separately licensed image encoders are still required. In one [two-run validation](data/reusable-session-validation.json), both outputs matched an earlier reference byte for byte; one input and one machine do not establish general latency or fidelity.

In a [two-photo sampling experiment](data/sampling-comparison.json), shorter predictor-corrector runs differed from the original 50-step latent by 34%–86% in relative Euclidean distance. These are limited comparisons to the original model, not ground-truth objects; the standard 50-step sampler remains the default.

The `Q8_SAFE` export policy stores the denoiser and background remover at source precision, preserves the octree in 32-bit floating point, and uses GGUF eight-bit `Q8_0` blocks for eligible Gaussian-decoder matrices. `Q4_SAFE` applies the same preservation policy with four-bit `Q4_0` Gaussian matrices. The other exporter policies are experimental and require separate fidelity evaluation.

By default, quantized denoiser and background tensors are expanded when loaded into PyTorch. The optional `--packed-denoiser` flag loads supported denoiser linear weights directly in packed form and expands each weight for its matrix multiplication. This can reduce resident memory, but its non-fused multiplication can increase execution time. The `Q2_K`, `Q4_K`, and `Q6_K` export policies require `--ggml-lib` pointing to a compatible local GGML shared library: Georgi Gerganov's tensor library for machine learning. An authoritative expansion of the GGML name is not established here. These policies preserve the octree in 32-bit floating point and use mixed precision elsewhere; the tier name does not mean every tensor uses that width. These policies are uncalibrated experiments, not quality guarantees.

The experimental `--fused-kernels` option implies `--packed-denoiser` and uses Triton on CUDA for supported `Q4_0`, `Q4_K`, and `Q8_0` linear matrices with float16 or bfloat16 activations. Most supported matrices are decoded inside the multiplication. Large `Q8_0` float16 multiplications use a temporary expanded matrix for a faster dense multiplication; packed weights remain resident and the temporary allocation increases peak memory. Unsupported encodings, float32 activations, and gradient-enabled calls retain the expansion path. Triton must be installed, and the first invocation compiles kernels; numerical results can differ from the expanded PyTorch path. On Windows, see the [Triton Windows installation and compatibility instructions](https://github.com/triton-lang/triton-windows); the reported workstation tests used PyTorch 2.10.0 with `triton-windows` 3.5.0.post21. Full-pipeline fidelity, memory use, and latency must be measured for the chosen model and hardware.

The experimental `Q8_MIXED` policy quantizes eligible matrices in denoiser blocks 12–23 and the Gaussian decoder, while preserving other denoiser tensors and the background remover at source precision. Its allocation comes from limited sensitivity experiments and needs broader validation before use as a release recommendation.

The mixed-precision `RUNTIME_F16_Q8` policy stores eligible Gaussian matrices in eight-bit blocks and restricts half-precision denoiser storage to block and refiner matrices that the pinned upstream runtime already converts to half precision. It leaves the other denoiser matrices at source precision. This aims to reduce storage without adding that conversion to the runtime; exact behavior depends on the runtime and must be validated when it changes.

Experimental `RUNTIME_F16_GAUSSIAN_Q2K`, `RUNTIME_F16_GAUSSIAN_Q4K`, and `RUNTIME_F16_GAUSSIAN_Q6K` exports use the same denoiser policy and two-bit, four-bit, or six-bit K-block Gaussian matrices. They require `--ggml-lib` and `--gaussian-calibration`: a plan created by `tools/create_gaussian_calibration.py` from recorded input moments and a report produced by `tools/select_gaussian_calibration.py`. The plan binds the original Gaussian weights and records the input artifact digests; those digests do not establish that the calibration or selection is representative. The native decoder currently expands these K-block Gaussian weights during cache creation; the experimental GPU path keeps supported linear weights packed between calls. These are mixed-precision models, not uniformly two-bit, four-bit, or six-bit models, and remain under evaluation.

For a local four-bit Gaussian experiment, provide a directory of distinct `.f32` Idol latents, an original Idol checkpoint, the matching SDK checkout, and a compatible local GGML library. The first sorted latent is held out; the next two are calibration inputs. More diverse held-out inputs are needed to assess quality.

```bash
python tools/calibrate_gaussian.py --weights /path/to/idol-source --sdk /path/to/idol-sdk \
  --latents /path/to/latents --out-dir calibration --holdout 1 --calib 2 --anchors 256
python tools/select_gaussian_calibration.py --weights /path/to/idol-source --sdk /path/to/idol-sdk \
  --latents /path/to/latents --moments calibration/hessians.pt --moments-report calibration/report.json \
  --type Q4_K --ggml-lib /path/to/libggml-base.so --out selection.json --holdout 1 --calib 2
python tools/create_gaussian_calibration.py --source /path/to/idol-source/decoder_gs.safetensors \
  --moments calibration/hessians.pt --moments-report calibration/report.json --selection selection.json \
  --type Q4_K --out gaussian-plan.npz
python tools/export_full_gguf.py --source /path/to/idol-source --out Idol-experimental.gguf \
  --tier RUNTIME_F16_GAUSSIAN_Q4K --ggml-lib /path/to/libggml-base.so \
  --gaussian-calibration gaussian-plan.npz --revision SOURCE_REVISION --report export-report.json
```

The revision argument is a claim about the checkpoint; verify it against the source. Replace `Q4_K` and the tier with the corresponding two-bit or six-bit names to run those experiments. The plan identifies the selection report but does not independently verify its derivation.

An alternative experimental workflow uses the full recorded covariance matrices to compensate quantization error between successive K-blocks. Using the `calibration` directory created above:

```bash
python tools/calibrate_gaussian_covariance.py --source /path/to/idol-source/decoder_gs.safetensors \
  --moments calibration/hessians.pt --moments-report calibration/report.json \
  --ggml-lib /path/to/libggml-base.so --type Q2_K --out gaussian-codes.npz
python tools/export_full_gguf.py --source /path/to/idol-source --out Idol-covariance-experimental.gguf \
  --tier RUNTIME_F16_GAUSSIAN_Q2K_COV --gaussian-encoded gaussian-codes.npz \
  --revision SOURCE_REVISION --report export-report.json
```

Use `--device cuda` during covariance encoding only when a suitable PyTorch installation and GPU are available. Add `--importance` to weight each quantization block by the covariance diagonal; this is a separate experimental choice that requires its own evaluation. Four-bit and six-bit codes can be produced by selecting `Q4_K` or `Q6_K` and the matching `RUNTIME_F16_GAUSSIAN_Q4K_COV` or `RUNTIME_F16_GAUSSIAN_Q6K_COV` tier. The encoder requires source-bound covariance data; the exporter checks the archive's source identity, tensor names, dimensions, quantization type and finite decoded values. The archive records provenance, but those identifiers do not independently prove the calculation or the representativeness of its inputs. The file remains mixed precision and requires separate evaluation for each type and workload.

In [two development-photo comparisons](data/gaussian-q2-covariance.json), covariance-compensated two-bit Gaussian matrices reduced mean foreground-color difference across six approximate views from 3.10% to 2.44% and from 2.95% to 2.05% compared with the earlier selected two-bit matrices. At 262,144 points, center-cell overlap changed from 90.79% to 90.75% on one image and from 83.35% to 85.11% on the other. Both variants used the same original-model latent and octree anchors. These are limited comparisons against the original decoder, not ground-truth reconstruction quality or evidence that every image improves.

For an experimental mixed two-/four-bit Gaussian allocation, create a second covariance archive with `--type Q4_K --out gaussian-q4-codes.npz`, then select and combine groups:

```bash
python tools/select_gaussian_mixed.py --weights /path/to/idol-source --sdk /path/to/idol-sdk \
  --q2-codes gaussian-codes.npz --q4-codes gaussian-q4-codes.npz \
  --train-latent train-1.f32 --train-latent train-2.f32 --eval-latent evaluation.f32 \
  --max-groups 8 --out gaussian-mixed-selection.json
python tools/create_mixed_gaussian_codes.py --source /path/to/idol-source/decoder_gs.safetensors \
  --q2 gaussian-codes.npz --q4 gaussian-q4-codes.npz --selection gaussian-mixed-selection.json \
  --groups 8 --out gaussian-mixed-codes.npz
python tools/export_full_gguf.py --source /path/to/idol-source --out Idol-mixed-experimental.gguf \
  --tier RUNTIME_F16_GAUSSIAN_Q2Q4_COV --gaussian-encoded gaussian-mixed-codes.npz \
  --revision SOURCE_REVISION --report mixed-report.json
```

The selector chooses groups from the training latents and separately reports evaluation latents; it rejects duplicate latent values even across different file formats. If selection stops before eight groups, set `--groups` no higher than the number actually selected. Deciding the group count after inspecting those evaluation results makes them development data, so use additional untouched inputs for a final assessment. The mixed archive records its per-matrix formats and source identifiers; the exporter validates these before writing.

In one [development experiment](data/gaussian-q2q4-mixed.json), 38 of 98 eligible Gaussian matrices used four bits and the others used two bits. With identical original latents and anchors, center-cell overlap improved from 90.8% to 95.4% and from 85.1% to 90.4% across two photographs compared with the two-bit covariance candidate. Mean foreground-color difference across six approximate views fell from 2.44% to 1.29% and from 2.05% to 0.95%. These photographs influenced development choices; the results do not establish performance on other images or against ground-truth objects.

To experiment with denoiser calibration, first export an original-runtime denoiser using `--tier RUNTIME_F16_Q8`. Supply local snapshots of both separately licensed encoders and distinct calibration images; reserve different images for evaluation. The collector compares every loaded denoiser value with the original source at the runtime's precision before recording covariance matrices.

```bash
python tools/export_full_gguf.py --source /path/to/idol-source --out Idol-runtime-f16.gguf \
  --tier RUNTIME_F16_Q8 --revision SOURCE_REVISION --report runtime-report.json
python tools/calibrate_denoiser.py --model Idol-runtime-f16.gguf \
  --denoiser-source /path/to/idol-source/denoiser.safetensors --sdk /path/to/idol-sdk \
  --dino /path/to/dinov3-snapshot --flux-vae /path/to/flux2-snapshot \
  --real-image calibration-1.png --real-image calibration-2.png --out-dir denoiser-calibration
python tools/calibrate_denoiser_covariance.py --source /path/to/idol-source/denoiser.safetensors \
  --moments denoiser-calibration/hessians.pt --moments-report denoiser-calibration/report.json \
  --ggml-lib /path/to/libggml-base.so --type Q8_0 --out denoiser-codes.npz
python tools/export_full_gguf.py --source /path/to/idol-source --out Idol-combined-experimental.gguf \
  --tier RUNTIME_F16_GAUSSIAN_Q2K_COV --gaussian-encoded gaussian-codes.npz \
  --denoiser-encoded denoiser-codes.npz --revision SOURCE_REVISION --report combined-report.json
```

The denoiser encoder also accepts `Q2_K`, `Q4_K`, and `Q6_K`; those names describe eligible denoiser block matrices, not the complete file. Its optional `--importance` flag weights each quantization block by the covariance diagonal and requires separate evaluation. The file contains both sets of encoded weights, original-precision octree weights, runtime-precision refiners and other tensors. Input hashes record the selected source files and images, but neither prove the calibration process nor guarantee accuracy on new inputs.

A [development comparison](data/denoiser-calibration-comparison.json) of 10-step calibration with 64 sampled tokens per forward against 50-step calibration with 16 sampled tokens per forward found image-dependent improvements and regressions in four- and six-bit denoisers. Because both the step count and sampling density changed, the results cannot isolate the effect of either choice. The measurements compare denoiser latents with the original model, not with ground-truth objects.

For a mixed denoiser that retains the most sensitive blocks at original runtime precision, first rank one quantized block at a time on separate calibration images, then combine that ranking with an encoded denoiser archive:

```bash
python tools/rank_denoiser_blocks.py --model Idol-runtime-f16.gguf \
  --denoiser-source /path/to/idol-source/denoiser.safetensors --sdk /path/to/idol-sdk \
  --dino /path/to/dinov3-snapshot --flux-vae /path/to/flux2-snapshot \
  --synthetic-image example-1.webp --synthetic-image example-2.webp \
  --ggml-lib /path/to/libggml-base.so --type Q4_K --out denoiser-ranking.json
python tools/create_mixed_denoiser_codes.py --source /path/to/idol-source/denoiser.safetensors \
  --encoded denoiser-codes.npz --ranking denoiser-ranking.json --blocks 8 --out denoiser-mixed-codes.npz
```

Pass `--denoiser-encoded denoiser-mixed-codes.npz` to `tools/export_full_gguf.py` with a runtime-precision base tier, including the mixed Gaussian tier. To rank the actual encoded matrices instead of quantizing blocks on demand, add `--encoded denoiser-codes.npz` to the ranking command and set `--type` to the archive's type. Ranking uses the worst reported calibration-image error per block; this is a heuristic, and the ranking format does not independently prove how scores were derived. The loader checks that the recorded quantized and preserved blocks match the current runtime's expected precision.

Experimental `--dino-gguf` and `--flux-vae-gguf` options accept separately obtained encoder files whose tensor names and shapes match the original encoder modules. For the supported DINOv3 file, the loader reads the architecture settings from its metadata; for the FLUX.2 autoencoder, it downloads a pinned public configuration and checks its digest. Explicit configuration overrides can be supplied with `--dino` and `--flux-vae`. File compatibility is not a claim of acceptable reconstruction quality or compatibility with another application's loader.

The tested external files include the [DINOv3 H+/16 vision transformer (ViT) Q4_0 conversion](https://huggingface.co/zisisnotzis/dinov3-gguf/blob/main/dinov3-vith16plus-pretrain-lvd1689m-q4_0.gguf), whose H+ variant uses 16-by-16-pixel image patches, and the [FLUX.2 autoencoder in mixed 32-bit and 16-bit storage](https://huggingface.co/gguf-org/flux2-dev-gguf/blob/main/pig_flux2_vae_fp32-f16.gguf). They are separate from the Idol model and remain subject to their sources' terms. The latter is an autoencoder file; the much larger quantized FLUX.2 diffusion models are not substitutes for it.

In [limited development comparisons](data/dino-precision-comparison.json), that third-party four-bit DINOv3 file used 476,860,704 bytes and produced 21.5% and 77.1% occupied-center-cell overlap at 32 cells per model-coordinate unit against the original encoder on two photographs using the same downstream decoder. These inputs influenced development, the comparison is against the original model rather than ground-truth objects, and the results do not guarantee quality on other images.

For an experimental graphics processing unit (GPU) decoding path, add `--decoder-backend gpu`. This uses the upstream Idol software development kit's mixed-precision octree and Gaussian decoders with packed supported Gaussian linear weights; it requires an NVIDIA Compute Unified Device Architecture (CUDA) device. Its sampling generator and numerical precision differ from the native CPU decoder, so matching seeds do not imply identical point clouds. The native decoder remains the default.

In a [comparison on three saved latents](data/decoder-cpu-vs-gpu.json) of the published model on an RTX 4090 and a 32-thread processor, the two decoders produced meshes that agree with original Idol equally well (mean mesh distance 0.064% to 0.088% of the bounding-box diagonal for either decoder, and 0.037% to 0.053% between the two). The GPU decoders took 0.3 to 1.2 s and the native decoder 37 to 62 s (8 and 4 threads), about 30 to 180 times longer. The native decoder peaked at 818 to 823 MiB of host memory; the GPU decoders at 523 MiB of PyTorch-allocated device memory, 305 MiB of it weights. This is three objects at one seed on a shared workstation, so treat the speed ratio as an order of magnitude.

On a CUDA device, `--stage-encoders` temporarily moves the denoiser to system memory during image preprocessing and encoding, then moves the image models to system memory before denoising. This can reduce peak GPU allocation at the cost of additional transfers and host memory; its effect depends on the device and input. It is a no-op on CPU. The pipeline is still built with every model on the GPU, so the build can set the peak. `tools/idol_generate_staged.py` takes the same arguments, builds every model on the CPU, and runs the two image encoders one after the other; the server does the same unless `--resident-gpu` is set. `tools/idol_staging.py` holds the implementation and is kept separate because the published benchmark receipts bind the exact bytes of `idol_generate.py` and `idol_session.py`. See the [measurements](data/idol-staged-memory.json).

`tools/idol_generate_lean.py` takes the same arguments (except `--packed-denoiser` and `--fused-kernels`) and keeps the models that are waiting on disk instead of in memory, as the server's lean mode does; `tools/idol_lowram.py` holds the implementation, apart from `idol_generate.py` and `idol_session.py` for the same reason. Each model is written once, as a safetensors file in a temporary directory removed on exit, and its module is moved to the `meta` device; `module.to(device)` reads the file back in one pass through two pinned buffers, and `module.to('cpu')` drops the tensors again. It changes only where the bytes wait: the point cloud matched staged mode's byte for byte in every run. The [measurements](data/idol-lowram.json) record the peaks by phase, what remains in the process, the restore speed, and what was not established.

Pass `--latent-out latent.f32` to retain the generated latent for reproducible decoder comparisons. The latent is a raw little-endian 32-bit floating-point array ordered by token and channel.

## Evaluation and development

```bash
ctest --test-dir build --output-on-failure
```

Some tests need additional Python packages and an SDK checkout in `IDOL_SDK`; a skipped test is not a passing validation. The [decoder measurements](benchmarks.html) describe their inputs and limits. The tools also compare sampled octree cells, Gaussian centers and colors, approximate rendered views, and mesh surfaces; these are diagnostic measures, not ground-truth 3D quality scores.

For three upstream example images at 50 sampling steps and one seed, the eight-bit candidate's mean foreground color difference across six approximate views ranged from 0.076% to 0.115%; center-cell overlap ranged from 99.39% to 99.69%. The corresponding four-bit ranges were 0.854% to 1.036% and 92.47% to 95.76%. Both decoders used the same generated latent. The [eight-bit](data/full-pipeline-q8.json) and [four-bit](data/full-pipeline-q4.json) reports record case hashes and per-view measurements. These results do not establish quality on other inputs or against ground-truth objects.

To compare decoders on a retained latent, first export the original weights with `tools/export_idol.py` and convert a GGUF file with `tools/gguf_to_decoder.py`. Decode the same latent with each directory, using the same `--level`, `--seed` and `--gaussians` values and `IDOL_QUANT_KERNEL=float`. Compare the resulting files with `tools/compare_splats.py` and `tools/render_splats_preview.py`. Repeat with more inputs and seeds; the preview renderer is an approximation, and shared latents do not test differences in image preprocessing or denoising.

When the experiment establishes that both files contain the same ordered anchors and per-anchor Gaussian slots, add `--paired` to `tools/compare_splats.py` to report displacement, opacity, and color differences at matching indices. Equal point counts alone do not establish that correspondence; do not interpret the paired measurements when sampling or point ordering may differ.

In a [two-photo development comparison](data/gaussian-weighted-index-paired.json), the weighted mixed two-/four-bit Gaussian decoder reduced mean paired center displacement versus its weighted two-bit counterpart from 0.00265 to 0.00153 and from 0.00389 to 0.00205 model-coordinate units. Maximum displacement increased slightly on the first photo, so this is not a uniform improvement. The runs used the same latent, octree, seed, and output ordering, but did not independently record the anchor bytes; the pairing relies on the sampling implementation reproducing the same ordered anchors. These measurements compare against the original decoder, not ground-truth geometry.

In a separate [shared-anchor comparison](data/gaussian-q4-q6-shared-anchors.json) on those two development photographs, weighted four-bit Gaussian matrices gave 98.46% and 94.98% center-cell overlap, while weighted six-bit matrices gave 99.45% and 98.56%. Mean foreground-color differences across [six approximate views](data/gaussian-q4-q6-views.json) were 0.60% and 0.40% for four bits, and 0.15% and 0.11% for six bits. The original and candidate decoders used the same in-memory ordered anchors and original latents. Each exported full model was checked against the encoded Gaussian archive and preserved reference tensors. These are decoder comparisons with limited inputs; they do not measure denoiser changes or ground-truth object quality.

One [full image-to-point-cloud run](data/gaussian-weighted-mixed-full-run.json) with the weighted mixed two-/four-bit Gaussian model produced the original-runtime latent and the same point-cloud bytes as the expanded-weight experiment. It took 49.9 seconds after Python imports on an NVIDIA GeForce RTX 4090; PyTorch reported 3.56 billion bytes of peak allocated graphics memory. This single run is not a latency distribution, and the PyTorch counter excludes other graphics-memory users and host memory.

## Running every test in Docker

`docker build` runs the CPU test suite, including the K-block quantization tests, which use the GGML library the
image builds and exposes as `IDOL_GGML_LIB`. A build has no GPU, so the fused-kernel test is skipped there. On a machine
with an NVIDIA GPU and the container toolkit, run it in the finished image:

```bash
docker run --rm --gpus all -e IDOL_TEST_GPU=1 idol-cpp \
  ctest --test-dir build -R fused_packed_gpu --output-on-failure
```

Triton compiles a small CUDA helper on first use, so the container needs a writable `/root` and `/tmp`. A read-only
root file system must mount both as `tmpfs`.

The native library and this repository's source code are available under the [Apache License 2.0](../LICENSE). The upstream Idol model and its separately licensed dependencies have their own terms; consult their original repositories and the [third-party notices](../THIRD_PARTY_NOTICES.md).
