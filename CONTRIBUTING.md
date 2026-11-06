# Contributing

Thanks for looking at idol.cpp. This is a small project, so the process is short.

## Setting up

You need CMake 3.21 or newer, a C++17 compiler, and an x86-64 CPU with AVX2, FMA and F16C. For the parity test you also
need Python 3 with PyTorch and NumPy; without PyTorch that test reports as skipped.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Also run the tests in a Release build before sending a change to a kernel. Optimization can shift floating-point results
slightly, and the tests allow for that.

## What a change needs

- **Tests.** A bug fix needs a test that fails without the fix. A new kernel needs a naive reference in the test suite,
  written in double precision where practical.
- **Parity.** If you change a decoder, compare it against the upstream software development kit on a fresh export.
  If you change the export layout, update `docs/format.html` in the same change.
- **Docs.** A change to a public header updates its comment, and a change that alters behaviour users can see updates
  the relevant page in `docs/` and the changelog under *Unreleased*.
- **Scope.** Keep a change to one thing. Refactors that move code should not also change behavior.

## Style

Match the code around you. In practice that means:

- Four-space indentation, lines of about 120 columns at most, and braces on the same line.
- `#pragma once` in headers, and the public API under `namespace idol`. Internal helpers live in `idol::detail`.
- Comments explain *why* a choice was made, such as a reference's boundary rule or a precision trade-off. Do not restate
  what the code says.
- Errors are reported with exceptions from the standard library. The public API throws `std::invalid_argument` for bad
  arguments and `std::runtime_error` for a bad file.
- No new dependencies. OpenMP is the only optional one, and the library must still build without it.

## License

Project contributions are governed by the [Apache License 2.0](LICENSE), including its terms for submitted
contributions. Retain applicable third-party notices for any included material.
