## A collection of functions to allow spreading out load across GPUs on sparse matrix × dense vector.
using LinearAlgebra
struct OpsSplits{T}
    Ns::Int64
    arrs::Array
    split_list::Array{Int64, 1}
end

Base.eltype(x::OpsSplits{T}) where {T} = T

struct MDCuSplits{T}
    Ns::Int64
    arrs::Array
    split_list::Array{Int64, 1}
end
Base.eltype(x::MDCuSplits{T}) where {T} = T
Base.size(x::MDCuSplits) = sum(map(collect, map(size, x.arrs)))

struct MDCuCopy{T}
    Ns::Int64
    arrs::Array
    split_list::Array{Int64, 1}
end
Base.eltype(x::MDCuCopy{T}) where {T} = T

struct MDCuArray{T}
    arr::CuArray{T}
    work_space::Union{MDCuCopy{T}, Nothing}
end
Base.eltype(x::MDCuArray{T}) where {T} = T
Base.collect(x::MDCuArray) = collect(x.arr)
Base.view(x::MDCuArray, args...) = view(x.arr, args...)
function maybe_split_view(x::MDCuArray, args...; split_hint=nothing)
    if isnothing(split_hint)
        MDCuArray(view(x.arr, args...), nothing)
    else
        arr = view(x.arr, args...)
        MDCuArray(arr, _prepare_workspace(arr, split_hint))
    end
end
maybe_split_view(x::AbstractArray, args...; split_hint=nothing) = view(x, args...)
Base.size(x::MDCuArray) = size(x.arr)

function assignto!(dst::AbstractArray, src::AbstractArray)
    dst .= src
end
function assignto!(dst::MDCuArray, src::MDCuArray)
    dst.arr .= src.arr
end
function assignto!(dst::MDCuArray, src::AbstractArray)
    dst.arr .= src
end
function assignto!(dst::AbstractArray, src::MDCuArray)
    dst .= src.arr
end


function LinearAlgebra.Hermitian(x::MDCuSplits, args...)
    @warn "Hermitian is bypassed for MDCuSplits"
    return x # do nothing when asking to be Hermitian
end

function _create_UM_arr(a_h; split_hint=nothing)
    et = eltype(a_h)
    @assert isbitstype(et) "element type of a_h, $(et), is not bits type."
    buf_a = Mem.alloc(Mem.Unified, sizeof(a_h))
    a_d = unsafe_wrap(CuArray{et}, convert(CuPtr{et}, buf_a), size(a_h))
    copyto!(a_d, a_h)

    work_space = nothing
    if !isnothing(split_hint)
        work_space = _prepare_workspace(a_d, split_hint)
    end

    return MDCuArray(a_d, work_space)
end

function _broadcast_operator_splits(As::OpsSplits)
    Ns = length(collect(devices()))
    et = eltype(As)
    @assert As.Ns==Ns "length of As is $(As.Ns); does not match Ns=$(Ns) that is given by device number."

    Ascu = Array{CUSPARSE.CuSparseMatrixCSC{et}}(undef, Ns)
    for (gpu, dev) in enumerate(devices())
        device!(dev)
        Ascu[gpu] = As.arrs[gpu]
    end
    return MDCuSplits{et}(Ns, Ascu, As.split_list)
end

_prepare_workspace(arr_out::MDCuArray, As) = _prepare_workspace(arr_out.arr, As)
function _prepare_workspace(arr_out::CuArray, As::Union{OpsSplits, MDCuSplits, Array{Int64, 1}})
    if typeof(As) <: Array
        split_list = As
    else
        split_list = As.split_list
    end
    et = eltype(arr_out)

    Ns = length(collect(devices()))
    arr_w = Array{CuArray{et}}(undef, Ns)

    @debug "$(size(arr_out))"
    for (gpu, dev) in enumerate(devices())
        device!(dev)
        arr_w[gpu] = CUDA.zeros(et, split_list[gpu], size(arr_out, 2))
    end

    for (gpu, dev) in enumerate(devices())
        device!(dev)
        synchronize()
    end
    return MDCuCopy{et}(Ns, arr_w, split_list)
end

function LinearAlgebra.mul!(arr_out::MDCuArray, Ascu::MDCuSplits, arr_in::MDCuArray, α::Number, β::Number; w::Union{MDCuCopy, Nothing}=nothing)

    if isnothing(w)
        if !isnothing(arr_out.work_space)
            w = arr_out.work_space
        else
            w = _prepare_workspace(arr_out.arr, Ascu)
        end
    end

    lb, ub = _get_lb_ub(Ascu.split_list)
    @. arr_out.arr = 0
    @sync for (gpu, dev) in enumerate(devices())
        device!(dev)
        mul!(w.arrs[gpu], Ascu.arrs[gpu], arr_in.arr, α, β)
    end


    for (gpu, dev) in enumerate(devices())
        device!(dev)
        @. arr_out.arr[lb[gpu]:ub[gpu], :] = w.arrs[gpu]
    end
    return nothing
end

function _split_sparse_matrix(A; Ns=2)
    # note: this is most naive splitting that may not be very efficient
    @info "Ns=$(Ns). Make sure to start main calculation with $(Ns) GPUs."
    et = eltype(A)
    As = Array{SparseMatrixCSC{et}}(undef, Ns)


    split_list = _partition_l(size(A, 1), Ns)
    lb, ub = _get_lb_ub(split_list)

    for i = 1:Ns
        # set column is faster; but we want non square. Warning: Hermitian is lost! This splitting need to be taken care of inside make_operators.jl
        As[i] = A[lb[i]:ub[i], :]
    end
    return OpsSplits{et}(Ns, As, split_list)
end

