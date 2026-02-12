# Unit-Test Ideas and Fine-Grained Improvement Backlog

## Immediate lightweight tests (no GPU runtime required)
- Confirm base package loads and `whichcore() == false` by default.
- Confirm `maybe_to_device`/`maybe_to_host` are identity on CPU arrays/sparse arrays.
- Confirm `maybe_on_device_zeros` returns regular `Array` without CUDA.
- Assert `Project.toml` contains `weakdeps`/`extensions` entries for CUDA.

## Conditional tests (run when CUDA is installed)
- Extension load test: import CUDA, then `using KPM`, verify CUDA-specialized methods dispatch on `CuArray`.
- Tiny CuArray conversion roundtrip: `Array -> maybe_to_device -> maybe_to_host`.
- Smoke-test one GPU kernel path at tiny dimensions (e.g., Chebyshev `N=4`) with correctness tolerance.

## Improvement ideas
- Replace explicit `Array` signatures with `AbstractArray` where safe to reduce duplication.
- Add `@testset` guards for optional deps using `Base.find_package("CUDA")`.
- Add dedicated CI job with CUDA dependency available (no required GPU execution).
- Add docs section: “How optional CUDA extension works”.
- Add benchmark script comparing CPU fallback vs extension-enabled GPU for tiny and medium cases.
