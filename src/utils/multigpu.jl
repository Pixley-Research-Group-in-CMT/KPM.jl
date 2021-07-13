## A collection of functions to allow spreading out load across GPUs on sparse matrix × dense vector.

struct OpsSplits{T}
    Ns::Int64
    arrs::Array
end

Base.eltype(x::OpsSplits{T}) where {T} = T

struct MDCuSplits{T}
    Ns::Int64
    arrs::Array
end
Base.eltype(x::MDCuSplits{T}) where {T} = T

struct MDCuCopy{T}
    Ns::Int64
    arrs::Array
end
Base.eltype(x::MDCuCopy{T}) where {T} = T

function _create_UM_arr(a_h)
    et = eltype(a_h)
    @assert isbitstype(et) "element type of a_h, $(et), is not bits type."
    buf_a = Mem.alloc(Mem.Unified, sizeof(a_h))
    a_d = unsafe_wrap(CuArray{et}, convert(CuPtr{et}, buf_a), size(a_h))
    copyto!(a_d, a_h)
    return a_d
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
    return MDCuSplits{et}(Ns, Ascu)
end

function _prepare_workspace(arr_out::CuArray)
    et = eltype(arr_out)

    Ns = length(collect(devices()))
    arr_w = Array{CuArray{et}}(undef, Ns)

    for (gpu, dev) in enumerate(devices())
        device!(dev)
        arr_w[gpu] = CUDA.zeros(et, N, NR)
    end

    for (gpu, dev) in enumerate(devices())
        device!(dev)
        synchronize()
    end
    return MDCuCopy{et}(Ns, arr_w)
end

function f!(Ascu::MDCuSplits, arr_in::CuArray, arr_out::CuArray; w::MDCuCopy=_prepare_workspace())
    @. arr_out = 0
    @sync for (gpu, dev) in enumerate(devices())
        device!(dev)
        mul!(w.arrs[gpu], Ascu.arrs[gpu], arr_in)
    end

    for (gpu, dev) in enumerate(devices())
        device!(dev)
        @sync @. arr_out += w.arrs[gpu]
    end
    return nothing
end

function _split_sparse_matrix(A; Ns=2)
    # note: this is most naive splitting that may not be very efficient
    @info "Ns=$(Ns). Make sure to start main calculation with $(Ns) GPUs."
    et = eltype(A)
    As = Array{SparseMatrixCSC{et}}(undef, Ns)

    if ishermitian(A)
        @assert (A.uplo == 'U') "only implemented for upper triangular representation of Hermitian"
        A = sparse(UpperTriangular(A))
    end

    for i = 1:Ns
        A_tmp = copy(A)
        A_tmp[:, i:Ns:end] .= 0 # set column is faster. Warning: Hermitian is lost! This splitting need to be taken care of inside make_operators.jl
        dropzeros!(A_tmp)

        if ishermitian(A)
            A_tmp = Hermitian(A_tmp, :U)
        end
        As[i] = A_tmp
    end
    return OpsSplits{et}(Ns, As)
end
