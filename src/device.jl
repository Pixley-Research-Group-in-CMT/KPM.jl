using Random
using SparseArrays
using LinearAlgebra

function whichcore()
    println("KPM.jl uses CPU only. GPU support is not added yet.")
end

maybe_to_device(x::SparseMatrixCSC) = x
maybe_to_device(x::Array) = x
maybe_to_device(x::SubArray) = x # Pushing SubArray to GPU should never happen

maybe_on_device_rand(args...) = rand(args...)
maybe_on_device_zeros(args...) = zeros(args...)

on_host_zeros(args...) = zeros(args...)

maybe_to_host(x::Array) = x
maybe_to_host(x::SparseMatrixCSC) = x
