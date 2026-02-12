# CUDA Optional-Dependency Migration: Self-Review (Pass 1)

## What is good
- Switched `CUDA` from hard dependency to package extension model:
  - `Project.toml`: added `[weakdeps]` and `[extensions]` (`KPMCUDAExt = "CUDA"`).
  - New `ext/KPMCUDAExt.jl` holds CUDA-specific code paths.
- CPU default path is now CUDA-free in `src/`:
  - Removed `using CUDA` in core source files.
  - Removed direct `CuArray`/`@cuda` references from default-loaded files.
- Preserved core behavior for CPU users via no-op device helpers in `src/device.jl`.
- Added lightweight tests (`test/optional_cuda_test.jl`) to guard CPU fallback behavior.

## Weak / risky points
- Could not run Julia tests locally (`julia` binary unavailable), so validation is static-only.
- Extension overload coverage is broad but not exhaustively verified for method ambiguity.
- GPU code in extension was migrated from legacy paths with minimal cleanup; potential latent runtime issues remain.

## Concrete fixes applied in pass 2
- Restored key CUDA-specialized routines in extension to avoid accidental GPU regression:
  - device conversion helpers
  - Chebyshev GPU kernels
  - conductivity helper kernels
  - selected vector/dot GPU overloads
- Kept CPU-first implementation simple and stable.

## Remaining follow-up suggestions
1. Add CI matrix leg that runs with CUDA dependency present (even if GPU unavailable) to ensure extension precompiles.
2. Add a tiny extension load test that conditionally executes when CUDA is installed.
3. Reduce duplication by moving shared formulas into CPU-generic helpers, keeping extension only for kernel launch wrappers.
