# Profiling Guide

This folder contains quick commands to profile kernels with Nsight Compute and Nsight Systems.

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

## Nsight Compute (Kernel metrics)

Vector Add:

```bash
ncu --set full --target-processes all build\vector_add.exe
```

Reduction:

```bash
ncu --set full --target-processes all build\reduction.exe
```

MatMul:

```bash
ncu --set full --target-processes all build\matmul.exe
```

Suggested metrics to inspect:
- Occupancy (sm__warps_active.avg.pct_of_peak_sustained_active)
- Memory throughput (dram__throughput.avg.pct_of_peak_sustained_elapsed)
- L2 hit rate (lts__t_sectors_hit_rate.pct)
- Warp stall reasons (smsp__warp_issue_stalled_*)

## Nsight Systems (Timeline)

```bash
nsys profile --stats=true -o nsys_report build\matmul.exe
```

Open the report:

```bash
nsys-ui nsys_report.qdrep
```

## Tips
- Use smaller input sizes when first validating correctness.
- Compare kernel time changes after adjusting tile size or block size.
