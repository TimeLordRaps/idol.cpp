# Third-party notices

The idol.cpp project source is available under the [Apache License 2.0](LICENSE).
The following third-party components and assets retain their own notices and terms.

The generated marching-cubes lookup tables in `src/mc_tables.h` are derived from
scikit-image, distributed under the BSD 3-Clause license. The notice also appears
in [the marching-cubes notice](https://github.com/TimeLordRaps/idol.cpp/blob/main/docs/notice.html):

```text
Copyright (c) the scikit-image developers. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

The original Idol model card in `UPSTREAM_README.md` and the accompanying
gallery assets come from [SperiD Labs' Idol model](https://huggingface.co/speridlabs/idol),
which is distributed under Apache License 2.0. A copy of the license is in
[`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt). The separately downloaded
Idol weights and image-encoder dependencies retain their respective licenses.

Optional prompt generation uses separately obtained components. The tested
[Iris-3B quantization](https://huggingface.co/webmp3/Sakura-Iris-3B-GGUF) and
[Qwen3-VL-4B-Instruct quantization](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF)
are marked Apache 2.0 by their respective model repositories. The tested
[stable-diffusion.cpp runner](https://github.com/leejet/stable-diffusion.cpp/blob/f5df5f8/LICENSE)
is distributed separately under the MIT license. Consult each source for its
license and distribution terms.

The 32-object benchmark uses Google Scanned Objects thumbnails distributed by
[`codepawl/scanned-objects-glb`](https://huggingface.co/datasets/codepawl/scanned-objects-glb)
at revision `fdb214f6ea18996f2d45e05df74c5d9696c3f7a2`. Original scans and images:
Google Research, copyright 2020 Google LLC, under
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
Thumbnail conversions are by codepawl. Input contact sheets resize and annotate
those thumbnails; reconstruction views are Idol model outputs. The benchmark
protocol preserves each original object source URL and input digest. See the
[Google Research dataset announcement](https://research.google/blog/scanned-objects-by-google-research-a-dataset-of-3d-scanned-common-household-items/).

`tools/idol_packed.py` includes block-decoding formulas adapted from gguf-py.

The Docker image also builds [GGML](https://github.com/ggml-org/ggml) v0.29.0 from source (MIT license, the
notice below) so the K-block quantization tests can run. It is not part of the repository source or the shipped model.

MIT License

Copyright (c) 2023 Georgi Gerganov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
