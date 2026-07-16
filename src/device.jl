using Random
using SparseArrays
using LinearAlgebra

"""
    AbstractDevice

Execution device for KPM workspaces. The base package provides `CPUDevice`;
the CUDA package extension (`ext/KPMCUDAExt.jl`) adds a GPU device and
activates it automatically when a functional GPU is present.
"""
abstract type AbstractDevice end

struct CPUDevice <: AbstractDevice end

# Set once by the CUDA extension's `__init__`; never mutated elsewhere.
const ACTIVE_DEVICE = Ref{AbstractDevice}(CPUDevice())

"""
    whichcore()

Return `true` when GPU support is active (CUDA loaded and a functional GPU
present), `false` for CPU-only operation.
"""
whichcore() = !(ACTIVE_DEVICE[] isa CPUDevice)

"""
    maybe_to_device(x[, expect_eltype])

Move `x` to the active device; no-op on CPU. On GPU, sparse matrices are
converted to CSR storage with element type `expect_eltype`, so that `mul!`
against the complex Chebyshev block vectors is supported by CUSPARSE.
SubArrays are never moved.
"""
maybe_to_device(x, expect_eltype=eltype(x)) = to_device(ACTIVE_DEVICE[], x, expect_eltype)
maybe_to_device(x::SubArray, expect_eltype=eltype(x)) = x

# Generic fallback: anything without a device-specific method stays put.
to_device(::AbstractDevice, x, expect_eltype) = x

"""
    maybe_to_host(x)

Copy `x` back to host memory; no-op for host arrays and numbers.
"""
maybe_to_host(x::Union{Array, SparseMatrixCSC, SubArray, Number}) = x

"""
    maybe_on_device_zeros(args...)
    maybe_on_device_rand(args...)

Allocate like `zeros`/`rand`, on the active device.
"""
maybe_on_device_zeros(args...) = device_zeros(ACTIVE_DEVICE[], args...)
maybe_on_device_rand(args...) = device_rand(ACTIVE_DEVICE[], args...)

device_zeros(::CPUDevice, args...) = zeros(args...)
device_rand(::CPUDevice, args...) = rand(args...)

on_host_rand(args...) = rand(args...)
on_host_zeros(args...) = zeros(args...)
