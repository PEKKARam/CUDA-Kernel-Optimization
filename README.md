# CUDA Kernel Optimization

This project contains basic CUDA kernel implementations for learning the CUDA
execution model and thread organization. It includes:
- Vector Add
- Reduction
- Matrix Multiplication (shared memory tiling)

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

## Run

```bash
build\vector_add.exe
build\reduction.exe
build\matmul.exe
```

## Profiling

See [profiling/README.md](profiling/README.md) for Nsight Compute and Nsight
Systems commands and metrics guidance.

