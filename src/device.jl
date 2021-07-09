using Random
using SparseArrays
using LinearAlgebra
using CUDA
using Logging

function whichcore()
    if CUDA.has_cuda()
        println("GPU support for KPM.jl is experimental..")
        return true
    end
    return false
end
whichcore()


function maybe_to_device(x::Union{SparseMatrixCSC, CUSPARSE.CuSparseMatrixCSC}, expect_eltype=dt_num)
    if !(eltype(x) <: expect_eltype)
        @warn "element type $(eltype(x)) is not in expect_eltype=$(expect_eltype). Not casting, though."
    end

    if CUDA.has_cuda()
        if (typeof(x) <: CUDA.CUSPARSE.CuSparseMatrixCSC)
            return x
        else
            return CUDA.CUSPARSE.CuSparseMatrixCSC{eltype(x)}(x)
        end
    else
        return x
    end
end

function maybe_to_device(x::Union{Array, CuArray}, expect_eltype=dt_num)
    if !(eltype(x) <: expect_eltype)
        @warn "element type $(eltype(x)) is not in expect_eltype=$(expect_eltype). Not casting, though."
    end

    if CUDA.has_cuda()# && eltype(x).isbitstype
        if (typeof(x) <: CUDA.CuArray)
            return x
        else
            return CUDA.CuArray{eltype(x)}(x)
        end
    else
        return x
    end
end


maybe_to_device(x::SubArray, expect_eltype=dt_num) = x # Pushing SubArray to GPU is bad for current CUDA stack.

maybe_to_host(x::Array) = x
maybe_to_host(x::SparseMatrixCSC) = x
maybe_to_host(x::CuArray) = Array(x)
maybe_to_host(x::CUSPARSE.CuSparseMatrixCSC) = SparseMatrixCSC(x)
maybe_to_host(x::Number) = x

function maybe_on_device_rand(args...)
    if CUDA.has_cuda()
        return CUDA.rand(args...)
    else
        return rand(args...)
    end
end


function maybe_on_device_zeros(args...)
    if CUDA.has_cuda()
        return CUDA.zeros(args...)
    else
        return zeros(args...)
    end
end



on_host_rand(args...) = rand(args...)
on_host_zeros(args...) = zeros(args...)

