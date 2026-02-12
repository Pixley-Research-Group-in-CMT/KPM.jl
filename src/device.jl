using Random
using SparseArrays
using LinearAlgebra
using Logging

"""
Report whether GPU support is active.

Base package defaults to CPU-only behavior; CUDA-specific activation is provided
by the optional package extension in `ext/KPMCUDAExt.jl`.
"""
whichcore() = false

function maybe_to_device(x::SparseMatrixCSC, expect_eltype=dt_num)
    if !(eltype(x) <: expect_eltype)
        @warn "element type $(eltype(x)) is not in expect_eltype=$(expect_eltype). Not casting, though."
    end
    return x
end

function maybe_to_device(x::Array, expect_eltype=dt_num)
    if !(eltype(x) <: expect_eltype)
        @warn "element type $(eltype(x)) is not in expect_eltype=$(expect_eltype). Not casting, though."
    end
    return x
end

maybe_to_device(x::SubArray, expect_eltype=dt_num) = x

maybe_to_host(x::Array) = x
maybe_to_host(x::SparseMatrixCSC) = x
maybe_to_host(x::SubArray) = x
maybe_to_host(x::Number) = x

maybe_on_device_rand(args...) = rand(args...)
maybe_on_device_zeros(args...) = zeros(args...)

on_host_rand(args...) = rand(args...)
on_host_zeros(args...) = zeros(args...)
