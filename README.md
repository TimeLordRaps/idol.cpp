# idol.cpp

Reconstruct a three-dimensional Gaussian point cloud from an image with [Idol](https://huggingface.co/speridlabs/idol). The [Idol-GGUF model repository](https://huggingface.co/TimeLordRaps/Idol-GGUF) provides a complete mixed-precision Idol model; this project supplies native decoders, an inference server, and a browser interface.

Idol itself takes an image. For text prompts, an optional image generator first produces that image; each prompt starts a new reconstruction. The tested generator is **Iris-3B**, using **Qwen3-VL-4B-Instruct** for text encoding. Both are separate downloads with Apache 2.0 licenses. This is not currently a real-time pipeline: the measured prompt-to-result times are below.

The [original Idol project](https://speridlabs.com/research/idol) also demonstrates Iris-3B images as inputs. Whether Idol was trained on Iris-generated images is not established by that project page.

## Models and access

| Component | Distribution | Default |
|---|---|---|
| Idol denoiser, octree decoder, Gaussian decoder, and background remover | [Idol-Q8_0.gguf](https://huggingface.co/TimeLordRaps/Idol-GGUF) | Complete Idol model, 967 MB; eligible matrices use eight-bit blocks, with other tensors at their original or runtime precision. |
| Image encoders | [DINOv3](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m) and [FLUX.2-dev autoencoder](https://huggingface.co/black-forest-labs/FLUX.2-dev) | Original precision; both checkpoints are gated and have separate terms. Accept their terms and authenticate with Hugging Face. |
| Optional prompt-to-image generator | [Iris-3B Q4_K_M](https://huggingface.co/webmp3/Sakura-Iris-3B-GGUF) and [Qwen3-VL-4B-Instruct Q4_K_M](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF) | Configured separately; the two files total about 5.9 GB. The interface identifies the configured files. |

DINOv3 is the third generation of the self-distillation with no labels vision-model family. GGUF (Georgi Gerganov's Unified Format) is a tensor file format; `Q8_0` and `Q4_K_M` identify weight encodings, rather than a claim that every tensor has the same precision. The original Idol card is preserved in [UPSTREAM_README.md](UPSTREAM_README.md). The project source is available under the [Apache License 2.0](LICENSE); separately distributed components retain their own terms. See the [third-party notices](THIRD_PARTY_NOTICES.md).

## Install

Requirements: Python 3.10 or newer; CMake 3.21 or newer; a C++17 compiler; and an x86-64 processor supporting Advanced Vector Extensions 2 (AVX2), fused multiply-add (FMA), and 16-bit floating-point conversion (F16C). An NVIDIA graphics processing unit (GPU) with a compatible Compute Unified Device Architecture (CUDA) PyTorch installation is recommended. The native decoders run on the central processing unit (CPU).

From a fresh checkout, create and activate a virtual environment, then follow [PyTorch's installation instructions](https://pytorch.org/get-started/locally/) for your device. On Windows, use `.venv\Scripts\Activate.ps1` instead of the activation command shown.

```bash
python -m venv .venv
source .venv/bin/activate
git clone https://github.com/speridlabs/idol-sdk.git
git -C idol-sdk checkout 4702a7f00850e34b6e83a2170cc123c3e9570744
python -m pip install ./idol-sdk -r requirements-runtime.txt -r requirements-server.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
```

Authenticate with Hugging Face after obtaining access to the original image encoders, then start the local server:

```bash
python tools/idol_server.py --repo TimeLordRaps/Idol-GGUF --file Idol-Q8_0.gguf
```

Open **http://127.0.0.1:8127** to upload an image or use a webcam and inspect the generated point cloud. The first request loads the models; later requests reuse the session. `--preload`, `--port`, and `--output-dir` control loading and output. The default loopback address is local to your computer; the server has no authentication, so use an authenticated proxy if exposing it over a network.

**Memory use is automatic.** The server asks `nvidia-smi` (or PyTorch, if that is unavailable) how much graphics memory is free and picks the fastest configuration that fits. With at least 6,488 MiB free it keeps every model resident, with batched guidance and cuDNN attention on. With less it runs the models in stages, described below. The choice is made each time a session is built, printed to the server's error output, and shown as `memory_mode` by `/health`. Override it with `--memory-mode resident`, `staged` or `lean`, switch the speed options with `--batch-guidance` / `--no-batch-guidance` and `--cudnn-attention` / `--no-cudnn-attention`, or state how much memory the server may count on, instead of what is measured free, with `--vram-budget-gib N` (to leave room for other programs, or for a device that cannot be measured). Only NVIDIA devices are measured; without a measurement the server stages Idol, and the image generator keeps its full budget. The thresholds come from [measured peaks](docs/data/memory-tiers.json) on one RTX 4090 and keep a 512 MiB headroom, so a different driver or workload can still run short.

For the measured configuration with the eight-bit fused kernels, also install a compatible Triton package and add `--fused-kernels`. These options affect memory and floating-point results. The Windows tests used PyTorch 2.10.0 and [triton-windows](https://github.com/triton-lang/triton-windows) 3.5.0.post21. Some large eight-bit operations temporarily expand weights, increasing peak memory.

Staged mode runs Idol with the least graphics memory: the server uses it when less is free, and `python tools/idol_generate_staged.py` runs it from the command line with the arguments of `idol_generate.py`. Every model is then built on the CPU and moved onto the GPU only while its stage runs, and the two image encoders run one after the other. With the published `Idol-Q8_0.gguf` and the original encoders on an RTX 4090, the whole process peaked at 2,754 MiB with the native decoder (PyTorch allocated at most 1,955 MiB) and completed under a 2.5 GiB allocator cap; at a 2.0 GiB cap it ran out of memory. The previous construction needed a 3.5 GiB cap and ran out of memory at 3.25 GiB. The staged point clouds were byte-identical at caps of 6, 3.5 and 2.5 GiB, while the previous construction's point cloud at a 3.5 GiB cap differed from its own at 6 GiB, so results can depend on memory pressure. Every run used the published file, one image and one seed. Staged inference needs host memory for every model and is slower than resident inference. A full text-to-3D request with a stated 4 GiB budget (Idol staged, image generator at 3 GiB) completed on a 24 GB card with the server and the generator together peaking at 3,332 MiB; a run on an actual 4 GB card was not performed. See the [staged-memory record](docs/data/idol-staged-memory.json) and the [end-to-end record](docs/data/memory-tiers.json).

**Low host memory.** Staged mode still holds every waiting model in system memory, dequantized. On Windows its process peaked at 5,183 MiB of private working set, which is memory that is not a memory-mapped file, so the system cannot simply drop it and read it again. Lean mode (`--memory-mode lean`, or `python tools/idol_generate_lean.py` from the command line, which takes the arguments of `idol_generate.py`) writes each model to a temporary file as soon as it is built and reads it back only while its stage runs, so the host holds one model at a time. The same run peaked at 3,286 MiB, and its point cloud was byte-identical to the staged one (one image, two denoising steps). Lean mode needs about 3.7 GB of free disk space in the temporary directory (`--lean-directory` chooses another), reads those models back for every generation (3.5 s in a restore measurement on a busy graphics device), and supports neither `--packed-denoiser` nor `--fused-kernels`. In `auto`, it is chosen when graphics memory calls for staging, the computer has less than 6,207 MiB of free memory, and 4,728 MiB of disk space is free; the choice and the reason are printed, and `/health` reports `lean`. What is left, about 3.3 GB, is held by PyTorch, CUDA and the graphics driver rather than by the models (a bare PyTorch and CUDA process takes 0.7 GB), so a computer with 4 GB of memory in total is not expected to be enough once the operating system is counted. In a Linux container on the same computer with a hard memory limit and no swap, lean mode completed two generations at a 4 GB limit (the process's own memory peaked at 2.3 GB; the rest of the limit was reclaimable file cache) and at 6 GB, while staged mode was killed for running out of memory at 4 GB and during its second generation at 6 GB. That limit covers the process alone, not an operating system beside it, and the graphics side was a stated 4 GiB budget on a 24 GB card, so a real 4 GB machine is untested; a text prompt additionally needs the image generator, which is the larger host-memory user (see Text prompts below). See the [low-memory record](docs/data/idol-lowram.json).

## Text prompts

Install a compatible [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) `sd-cli` executable, then configure the tested Iris and Qwen files:

```bash
python tools/setup_iris.py --runner /path/to/sd-cli
```

The setup downloads pinned files, checks their Secure Hash Algorithm 256-bit (SHA-256) digests, and does not overwrite an existing configuration. Use `--image-model` and `--text-encoder` to reuse the tested files on disk. The tested runner was stable-diffusion.cpp `master-1052-f5df5f8`; other versions have not been established as compatible. The generator runs as its own process and exits before Idol loads. When at least 7,222 MiB of graphics memory remain free with the Idol session loaded, the server keeps the session and runs the generator beside it. With less, it releases the session before the prompt's image is generated and rebuilds it afterwards, which adds loading time (tens of seconds, and in lean mode the cache files are written again; a stated `--vram-budget-gib` always releases it).

The generator's graphics-memory budget is also chosen automatically for each image. With at least 7,222 MiB free it runs at its full size, every weight resident, which is fastest. With less it takes the largest budget that was measured to fit, down to 2 GiB; the runner then loads only the weights it needs and cuts the graph into segments, so the text encoder and the image model are not resident together. In runs on an RTX 4090 with the Vulkan runner, the generator's process peaked at 6,710 MiB at a budget of 16 or 8 GiB, about 4,400 MiB at 6, 5 or 4, 3,287 MiB at 3, and 2,272 MiB at 2, and a budget of 1 failed. The earlier quieter runs took about three to four times as long at 2 or 3 GiB as at 16. The images matched the 16 GiB image pixel for pixel at every budget for that one prompt. To fix the budget instead, add `--max-vram-gib N` to `setup_iris.py` or set `max_vram_gib` in the configuration; `auto` is the default. This covers the image generator only, not Idol; see the [memory-budget record](docs/data/iris-memory-budget.json) and [measured peaks](docs/data/memory-tiers.json). Its system memory is a separate matter: the more of its weights are kept off the graphics device, the more it holds in private host memory. Measured as private working set, the generator alone peaked at about 1.1 GB with every weight resident (a 16 GiB budget), 3.5 GB at 6 GiB and 6.7 to 6.9 GB at 4 GiB and below. A complete text-to-3D request with a stated 4 GiB budget (generator at 3 GiB) peaked at 7.0 GB of private host memory in both staged and lean mode, because the generator, not Idol, sets the peak while it runs, and it returned the same point cloud both times. The runner's `--mmap` option kept its private memory at 1.2 to 1.4 GB (2.4 GB at 2 GiB) and produced a byte-identical image, with the weights counted instead as mapped file pages that the system can drop. The server therefore passes `--mmap` when the free system memory is below what the generator held at the budget in use plus a 1,024 MiB headroom (set `"mmap": true` or `false` in the image-generator configuration, or `setup_iris.py --mmap auto|on|off`, to override); with mapping, the same request in lean mode peaked at 3.8 GB of private host memory instead of 7.0 GB and returned the same point cloud. The mapped pages are read back from disk when memory is short, so the speed cost on a computer that really is short of memory, and the whole path on one with 4 to 6 GB, were not tested. One prompt, one seed and one 64 GiB computer: see the [text-prompt record](docs/data/text-prompt-host-memory.json).

The browser displays the configured image-model and text-encoder filenames and the generated image. For an external service, its reported model identity is displayed; filenames and service labels alone do not independently verify the loaded weights. See [development and configuration details](docs/development.md) for optional image services and manual settings.

## Mesh generation and comparison

Open the **Mesh** tab after a reconstruction and choose **Generate mesh**. It previews an untextured triangle mesh and provides a Wavefront object (`.obj`) download. You can also upload a saved Gaussian point cloud (`.ply`) to extract its mesh without rerunning image inference. The server finds the built `idol_mesh` tool in `build/`, `build/Release/`, or the executable search path; use `--mesh-executable /path/to/idol_mesh` to select it explicitly.

Upload an original-model mesh under **Reference mesh**, then choose **Compare mesh with reference**. The displayed results and downloadable report compare sampled surface points with the other mesh's closest triangles, without alignment. The report records source and mesh digests, extraction settings when available, and the sampling seed. Uploaded reference extraction settings are unknown unless independently recorded; use identical settings on both sides.

The primary geometric fidelity measures are mean bidirectional distance, the larger directional 95th-percentile distance, and F1 scores. F1 is the harmonic mean of precision and recall within a distance threshold of 0.1%, 0.5%, or 1% of the reference bounding-box diagonal. Distances are in model-coordinate units, with no assumed physical scale. These are finite-sample geometry comparisons; mesh extraction omits Gaussian color, scale, and rotation information, so rendered-color comparisons remain complementary.

A fixed challenge cohort of 32 household objects and three matched seeds produced 96 original-versus-eight-bit comparisons. Each used original-precision encoders, 50 steps and 262,144 Gaussians. The table summarizes agreement with original Idol; it is **not ground-truth reconstruction accuracy**.

| Measurement | Mean | Median | Worst observed |
|---|---:|---:|---:|
| Mean mesh distance / reference diagonal | 0.124% | 0.089% | 1.715% |
| Mesh F1 within 0.1% of diagonal | 68.49% | 71.51% | 5.24% |
| Mesh F1 within 0.5% of diagonal | 96.74% | 98.92% | 24.78% |
| Approximate-view foreground color error | 2.22% | 1.68% | 28.59% |
| Approximate-view silhouette intersection over union | 98.63% | 99.38% | 64.70% |

The worst case was Lego Friends Mia at seed 2026; its difference is material and should not be dismissed by the average. The selected cohort is not a random sample. The mesh extractor uses Gaussian centers and opacity, with smoothing and component filtering that can hide differences; appearance is evaluated separately with an approximate renderer. See the [full protocol, results, figures, and limitations](docs/benchmark-gso32.md) and [mesh algorithm](docs/mesh.html).

The server also exposes `POST /v1/reconstructions/{result_id}/mesh` for an existing result, `POST /v1/meshes` for an uploaded Gaussian cloud, and `POST /v1/meshes/{mesh_id}/comparisons` with an uploaded reference mesh. See the [development guide](docs/development.md) for examples.

The broader fidelity cohort also timed model-resident generation on this shared workstation: median subsequent calls were **25.57 seconds original** and **28.82 seconds eight-bit** (64 calls each), with median peak PyTorch allocations of **5.37 GB** and **4.72 GB**, respectively. These are not load-inclusive or end-to-end timings; execution order was fixed. They show a lower allocation in this run, but do not establish a speed saving or its cause. The [cohort report](docs/benchmark-gso32.md#runtime-observations) includes first calls and 95th percentiles.

## Measured performance on one RTX 4090

These are individual runs on Windows with an NVIDIA GeForce RTX 4090 (24,564 MiB physical GPU memory), PyTorch 2.10.0 with CUDA 12.8, and `triton-windows` 3.5.0.post21. Image-to-3D runs used 50 denoising steps, seed 42, 262,144 Gaussian points, resident GPU models, batched guidance, optimized attention, and fused kernels. Each row identifies the tested inputs and scope; operating-system caches could be warm. The staged rows disable resident models, batched guidance, and optimized attention. The browser timing also includes browser launch and page loading.

| Workload | Model / image encoders | Elapsed time | Peak GPU allocation | Peak host working set |
|---|---|---:|---:|---:|
| Fresh process, image to 3D; lamp and axolotl, one run each | Original Idol / original precision | 37.0 s; 36.0 s | 4.76 GB | ~6.43 GB |
| Fresh process, image to 3D; same two images | Idol Q8 / original precision | 34.6 s; 34.6 s | 4.39 GB | ~6.73 GB |
| Warm request, image to 3D; lamp and axolotl, one run each | Original Idol / original precision | 25.12 s; 25.12 s | 5.36 GB | ~6.07 GB |
| Warm request, image to 3D; lamp and axolotl, one run each | Idol Q8 / original precision | 28.05 s; 27.38 s | 4.73 GB | ~6.72 GB |
| Warm request, image to 3D; axolotl | Idol Q8 / third-party DINOv3 Q4, original FLUX.2 autoencoder | 30.20 s | 3.51 GB | 3.63 GB |
| Warm staged request, image to 3D; axolotl | Idol Q8 / original encoders | 33.28 s | 3.51 GB | 6.72 GB |
| Warm staged request, image to 3D; axolotl | Idol Q8 / third-party DINOv3 Q4, original FLUX.2 autoencoder | 30.92 s | **2.31 GB** | 5.42 GB |

GPU values are peak **PyTorch allocated memory**, not total device use: they exclude the GPU context, reserved-but-unused allocations, the separate Iris process, and other applications. Host values are the Windows process peak working set. Decimal GB means one billion bytes; MiB means 1,048,576 bytes. A 16 GiB Iris configuration limit is a budget, not a measured peak. These observations do not establish a minimum GPU size, a latency distribution, or speed on another machine. Iris generation and Idol run sequentially, so the separate peaks must not simply be added.

The following two fresh-process prompt trials measure the whole workflow using different counters. Both use Iris-3B Q4 and Qwen3-VL-4B Q4 to generate a 1,024 × 1,024 image in 30 steps, followed by Idol Q8 with the original encoders. Browser launch and page loading are included in the time; starting the server beforehand is excluded.

| Trial | Browser launch to result | Peak total-device GPU use | Pre-request device use | Peak process-tree working-set sum | Peak process-tree private commit |
|---|---:|---:|---:|---:|---:|
| 1 | 100.16 s | 12.85 GB | 5.57–5.76 GB | 7.43 GB | 11.17 GB |
| 2 | 72.74 s | 13.12 GB | 5.56 GB | 6.87 GB | 11.96 GB |

Total-device graphics usage includes the display and other applications; it is not an isolated Idol allocation. Working-set sums can count shared pages more than once, and private commit includes allocations that may be paged out. The process tree includes the browser, image generator, server, and measurement process. Samples were requested every 0.25 seconds, with maximum observed gaps of 0.31 and 0.34 seconds, so shorter peaks may be missed. The [prompt measurement record](docs/data/prompt-memory-rtx4090.json) includes component hashes, settings, timing boundaries, and phase measurements. Two runs do not establish a minimum memory requirement or a latency distribution.

The Q4 DINOv3 result above is an **experimental compatibility measurement, not a quality recommendation**. On the axolotl image, with matched staged settings and the same Idol Q8 model, its latent differed by 34.17% in relative Euclidean norm from the original-encoder latent. In two image comparisons with original encoders, Idol Q8 latent differences were 3.23% and 6.47%. These measure agreement with Idol, not accuracy against ground-truth objects. These are individual observations; more images and seeds are needed before claiming a general quality or memory minimum. The [measurement record](docs/data/rtx4090-release-benchmark.json) binds the figures to input, model, output, and runtime source digests where recorded.

To test a separately obtained compatible image encoder, the server accepts `--dino-gguf /path/to/file.gguf` and `--flux-vae-gguf /path/to/file.gguf`. The measured DINO file was [dinov3-vith16plus-pretrain-lvd1689m-q4_0.gguf](https://huggingface.co/zisisnotzis/dinov3-gguf) at revision `7fbb86b66bbb1dd483c94c179d35c969c76467e6` and SHA-256 `cf80ee6e3c2d4044b3214c0ccde5e416cb999b799f35b699f4edaa4059d83d71`. Verify the separate source terms. A low-bit FLUX.2 autoencoder has not been validated here; a quantized FLUX.2 image-generation diffusion model is not a substitute for the autoencoder.

## Command line and application interface

```bash
python tools/idol_generate.py --repo TimeLordRaps/Idol-GGUF --file Idol-Q8_0.gguf \
  --image photo.jpg --mode real --out result.ply --decoder build/idol_decode
```

On Windows, use the path to `idol_decode.exe`. The output is a Polygon File Format (PLY) Gaussian point cloud, alongside the preprocessed input and predicted camera. Existing output files are not overwritten.

The application programming interface (API) uses the same port. `GET /health` reports loading status; `POST /v1/reconstructions` accepts a multipart image, and `POST /v1/text-to-3d` accepts a JSON prompt when an image generator is configured. Results include point-cloud bytes and metadata headers.

```bash
curl -F image=@photo.jpg http://127.0.0.1:8127/v1/reconstructions -o result.ply
curl -H 'Content-Type: application/json' -d '{"prompt":"a red chair"}' \
  http://127.0.0.1:8127/v1/text-to-3d -o prompt-result.ply
```

## Docker

The supplied Dockerfile pins the upstream Idol software development kit (SDK) and primary runtime dependencies. A locally built image, using cached intermediate build stages, passed installed-package integrity and import checks and reconstructed an image with the released Q8 model and original encoders. On an RTX 4090, that one 50-step request produced 262,144 finite points in **58.29 seconds**, including loading; mesh extraction produced 80,660 triangles. PyTorch reported **3.51 GB** peak allocated graphics memory, excluding the graphics context, reserved memory, and other processes. The [fresh-image test record](docs/data/docker-fresh-runtime-rtx4090.json) identifies the image, tested Dockerfile, model, and limitations.

The test used read-only mounts for the model and encoder cache. A separate headless Chromium check against the same image generated a new reconstruction, extracted a mesh, and compared a saved cloud with a reference mesh. An earlier explicitly software-rendered browser run timed out waiting for a mesh status update. These tests did not cover downloads, prompt image generation in Docker, or a build with an empty cache. A separate [GitHub-hosted Docker build](docs/data/docker-build-ci.json) passed its build checks without GPU inference. Building locally requires sufficient disk space, network access, compatible GPU access, and the separately licensed encoder weights.

```bash
docker build -t idol-cpp .
docker run --gpus all -p 127.0.0.1:8127:8127 -e HF_TOKEN \
  -v idol-hf-cache:/root/.cache/huggingface idol-cpp \
  python tools/idol_server.py --host 0.0.0.0 \
  --repo TimeLordRaps/Idol-GGUF --file Idol-Q8_0.gguf
```

Set `HF_TOKEN` in the host environment after obtaining access to both gated image encoders. Prompt generation additionally needs a Linux-compatible `sd-cli`, its model files, and a configuration with paths valid inside the container; these are not bundled.

To check a running server with a real image and save its point cloud and receipt, run the following in a Python environment with `requests`, NumPy, and `plyfile` installed. The output directory must be new.

```bash
python tools/verify_server_reconstruction.py --image assets/gallery/axolotl.webp --output /tmp/idol-smoke
```

## Development and evaluation

Run `ctest --test-dir build --output-on-failure` after building. Some tests require additional dependencies or the pinned SDK; a skipped test does not establish a pass. The [development guide](docs/development.md) retains experimental quantization workflows, diagnostics, and reusable-session details. Decoder-specific reports are in [benchmarks](docs/benchmarks.html). Approximate views and latent comparisons are diagnostics, not ground-truth measures of reconstruction quality.
