using Random
using SparseArrays
using LinearAlgebra
using CUDA

function whichcore()
    if CUDA.has_cuda()
        println("GPU support for KPM.jl is experimental..")
        return true
    end
    return false
end
whichcore()

function maybe_to_device(x::SparseMatrixCSC)
    if CUDA.has_cuda()
        return CUDA.CUSPARSE.CuSparseMatrixCSC(x)
    else
        return x
    end
end
maybe_to_device(x::CUSPARSE.CuSparseMatrixCSC) = x

function maybe_to_device(x::Array)
    if CUDA.has_cuda()# && eltype(x).isbitstype
        return CUDA.CuArray(x)
    else
        return x
    end
end
maybe_to_device(x::CuArray) = x

maybe_to_device(x::SubArray) = x # Pushing SubArray to GPU should never happen

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

