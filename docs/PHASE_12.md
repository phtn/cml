# Phase 12 — Performance and library readiness

Measure the numerical kernels on representative sizes and validate the results
before comparing runtime. Build the library for reuse and keep the normal
tests runnable through both Make and CMake.

## Build sequence

1. Block matrix multiplication so the working rows stay in cache. Compare
   against the direct row-column implementation and an optional CBLAS backend.
2. Add K-nearest neighbors and K-means to exercise the numerical and data
   primitives outside regression and neural networks. Use deterministic tie
   handling and test refitting and seeded cluster initialization.
3. Compile a static library, track header dependencies, run examples and
   integration tests, and test under address and undefined-behavior sanitizers.
4. Run the matrix and training benchmarks. Report input size, time, optimizer
   updates, and final loss. Optional BLAS comparisons use the same matrices
   and check the result against the direct reference computation.

Use `make all test example`, `make sanitize`, and `make benchmark`.
The Make and CMake options in the [README](../README.md) select an optional
OpenBLAS or Apple Accelerate backend. The remaining advanced algorithms in the
main guide are extension projects.

## Verification

The release build passed all 18 CTest suites. The Make suites also passed with
Clang, GCC, AddressSanitizer, and UndefinedBehaviorSanitizer. The model workflow
reported zero leaks in the macOS `leaks` run.

One local arm64 run used Apple Clang at `-O2` and the same 512 × 512 input for
each backend. Times are the best of three runs. Every result matched the
direct reference within the benchmark tolerance.

| Backend | Time |
| --- | ---: |
| Direct row-column reference | 142 ms |
| CML blocked C11 | 24.1 ms |
| Single-threaded OpenBLAS 0.3.30 | 5.44 ms |
| Apple Accelerate | 0.642 ms |

The training benchmark also completed its one-million-sample stress run for
full-batch, stochastic, and four mini-batch sizes.
[OpenBLAS release used for this comparison](https://github.com/OpenMathLib/OpenBLAS/releases/tag/v0.3.30)
