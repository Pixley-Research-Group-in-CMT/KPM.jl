using Random
using SparseArrays
using LinearAlgebra
using CUDA

function whichcore()
    println("KPM.jl uses CPU only. GPU support is not added yet.")
end
whichcore()

@generated function maybe_to_device(x::SparseMatrixCSC)
    if CUDA.has_cuda()
        return :(CUSPARSE.CuSparseMatrixCSC(x))
    else
        return :(x)
    end
end
maybe_to_device(x::CUSPARSE.CuSparseMatrixCSC) = x

@generated function maybe_to_device(x::Array)
    if CUDA.has_cuda()# && eltype(x).isbitstype
        return :(CuArray(x))
    else
        return :(x)
    end
end
maybe_to_device(x::CuArray) = x

maybe_to_device(x::SubArray) = x # Pushing SubArray to GPU should never happen

maybe_to_host(x::Array) = x
maybe_to_host(x::SparseMatrixCSC) = x
maybe_to_host(x::CuArray) = Array(x)
maybe_to_host(x::CUSPARSE.CuSparseMatrixCSC) = SparseMatrixCSC(x)


@generated function maybe_on_device_rand(args...)
    if CUDA.has_cuda()
        return :(CUDA.rand(args...))
    else
        return :(rand(args...))
    end
end


@generated function maybe_on_device_zeros(args...)
    if CUDA.has_cuda()
        return :(CUDA.zeros(args...))
    else
        return :(zeros(args...))
    end
end



on_host_rand(args...) = rand(args...)
on_host_zeros(args...) = zeros(args...)






function maybe_to_device_buffer(x::Array{T, 1} where {T <: SubArray}; buffer_size::Integer=3)
    if CUDA.has_cuda()
        println("creating buffered subarray")
        x1 = first(x)
        bsa_template = BufferedSubArray{CuArray}(x1;
                                                 buffer_size=buffer_size)
        return BufferedSubArray{CuArray}.(x; shared_with=bsa_template)
    else
        return BufferedSubArray{SubArray}(x)
    end
end


