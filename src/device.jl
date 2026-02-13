using Random
using SparseArrays
using LinearAlgebra
# using CUDA
# using CUDA.CUSPARSE
using Logging

# function whichcore()
#     if CUDA.has_cuda()
#         println("GPU support for KPM.jl is experimental..")
#         return true
#     end
#     return false
# end
# whichcore()

"""
Report whether GPU support is active.

Base package defaults to CPU-only behavior; CUDA-specific activation is provided
by the optional package extension in `ext/KPMCUDAExt.jl`.
"""
whichcore() = false


#function maybe_to_device(x::Union{SparseMatrixCSC, CuSparseMatrixCSC}, expect_eltype=dt_num)
function maybe_to_device(x::SparseMatrixCSC, expect_eltype=dt_num)
    if !(eltype(x) <: expect_eltype)
        @warn "element type $(eltype(x)) is not in expect_eltype=$(expect_eltype). Not casting, though."
    end

    # if CUDA.has_cuda()
    #     if (typeof(x) <: CuSparseMatrixCSC)
    #         return x
    #     else
    #         return CuSparseMatrixCSC{eltype(x)}(x)
    #     end
    # else
    #     return x
    # end
    return x
end

# function maybe_to_device(x::Union{Array, CuArray}, expect_eltype=dt_num)
function maybe_to_device(x::Array, expect_eltype=dt_num)
    if !(eltype(x) <: expect_eltype)
        @warn "element type $(eltype(x)) is not in expect_eltype=$(expect_eltype). Not casting, though."
    end

    # if CUDA.has_cuda()# && eltype(x).isbitstype
    #     if (typeof(x) <: CuArray)
    #         return x
    #     else
    #         return CuArray{eltype(x)}(x)
    #     end
    # else
    #     return x
    # end
    return x
end


maybe_to_device(x::SubArray, expect_eltype=dt_num) = x # Pushing SubArray to GPU is bad for current CUDA stack.

maybe_to_host(x::Array) = x
maybe_to_host(x::SparseMatrixCSC) = x
# maybe_to_host(x::CuArray) = Array(x)
# maybe_to_host(x::CuSparseMatrixCSC) = SparseMatrixCSC(x)
maybe_to_host(x::SubArray) = x
maybe_to_host(x::Number) = x

# function maybe_on_device_rand(args...)
#     if CUDA.has_cuda()
#         return CUDA.rand(args...)
#     else
#         return rand(args...)
#     end
# end


# function maybe_on_device_zeros(args...)
#     if CUDA.has_cuda()
#         return CUDA.zeros(args...)
#     else
#         return zeros(args...)
#     end
# end
maybe_on_device_rand(args...) = rand(args...)
maybe_on_device_zeros(args...) = zeros(args...)


on_host_rand(args...) = rand(args...)
on_host_zeros(args...) = zeros(args...)

