using Random
using SparseArrays
using LinearAlgebra

function whichcore()
    println("No CUDA branch..")
    return false
end
whichcore()

@generated function maybe_to_device(x::SparseMatrixCSC)
        return :(x)
end

@generated function maybe_to_device(x::Array)
        return :(x)
end

maybe_to_device(x::SubArray) = x # Pushing SubArray to GPU should never happen

maybe_to_host(x::Array) = x
maybe_to_host(x::SparseMatrixCSC) = x
maybe_to_host(x::Number) = x

@generated function maybe_on_device_rand(args...)
        return :(rand(args...))
end


@generated function maybe_on_device_zeros(args...)
        return :(zeros(args...))
end



on_host_rand(args...) = rand(args...)
on_host_zeros(args...) = zeros(args...)

