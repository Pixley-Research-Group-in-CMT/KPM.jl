#module BufferedSubArrays
using CUDA
export BufferedSubArray, load_buffer!, unload_buffer!, get_buffer, ischeckedout
struct BufferedSubArray{T<:Union{SubArray, CuArray}}
    # All operation interact with workspace array, `buffer`. 
    # Load and unload should be called upon to update the main array.
    # This data structure facilitates large array saved on CPU memory
    # with only a small portion loaded to GPU, for example. The GPU
    # memory [buffer] is pre-allocated and may be shared across multiple
    # instance of BufferedSubArray.
    #
    # When no GPU is available, using BufferedSubArray is similar (TODO make
    # similar into equivalent) to SubArray, while `load_buffer`
    # and `unload_buffer` are null operations, for convenience in
    # writing code that works for both environments.
    #
    # (potentially this can also be extended to the use case where
    # a large array is saved on disk).
    #
    # buffer_registry is shared. (TODO lock? currently only supporting serial.)
    #
    subarray::SubArray
    buffer::Array{T, 1}              ### SHARED for CuArray
    buffer_registry::Array{Bool, 1}  ### SHARED for CuArray
    buffer_idx::Array{Int64, 1}
    buffer_checkedout::Array{Bool, 1}


    function BufferedSubArray{T}(subarray::SubArray; buffer_size::Integer=3, shared_with::Union{BufferedSubArray, Nothing}=nothing) where {T}
        if T == SubArray
            return new{SubArray}(subarray, [subarray], [false], [1], [false])
        end
        if isnothing(shared_with)
            buffer = map(i -> CuArray(subarray), 1:buffer_size)
            buffer_registry = zeros(Bool, buffer_size)

        else
            buffer = shared_with.buffer
            buffer_registry = shared_with.buffer_registry
        end
        buffer_idx = [0]
        buffer_checkedout = [false]
        new{CuArray}(subarray, buffer, buffer_registry, buffer_idx, buffer_checkedout)
    end
end

# check if the subarray can be directly copied to a CuArray
direct_copiable(src::SubArray) = (src.stride1 == 1) && (src.offset1 == 0)

ischeckedout(bsa::BufferedSubArray) = bsa.buffer_checkedout[1]

buffer_available(bsa::BufferedSubArray) = bsa.buffer_registry[bsa.buffer_idx]


function unsafe_copy_to_device!(src::SubArray, dest::CuArray)
    if !direct_copiable(src)
        src = Array(src)
    end
    unsafe_copyto!(pointer(dest), pointer(src), length(src))
end

# Do nothing when both on CPU
function load_buffer!(bsa::BufferedSubArray{CuArray}; load_idx::Int64=findfirst(!, bsa.buffer_registry))
    # load data in subarray to buffer at given load_idx. If not given, find first available
    # WARNING: this will blindly trust the given load_idx to be non-conflicting
    unsafe_copy_to_device!(bsa.subarray, bsa.buffer[load_idx])

    bsa.buffer_registry[load_idx] = true
    bsa.buffer_idx[1] = load_idx
    bsa.buffer_checkedout[1] = true
    return nothing
end
function unload_buffer!(bsa::BufferedSubArray{CuArray})
    # write content in buffer back to main array.
    bsa.subarray .= Array(bsa.buffer[first(bsa.buffer_idx)])

    bsa.buffer_registry[bsa.buffer_idx[1]] = false
    bsa.buffer_idx[1] = 0
    bsa.buffer_checkedout[1] = false
    return nothing
end

function load_buffer!(bsa::BufferedSubArray{SubArray})
    bsa.buffer_checkedout[1] = true
    bsa.buffer_registry[1] = true
    return nothing
end
function unload_buffer!(bsa::BufferedSubArray{SubArray})
    bsa.buffer_checkedout[1] = false
    bsa.buffer_registry[1] = false
    return nothing
end

function get_buffer(bsa::BufferedSubArray)
    # return the array to work with
    @assert ischeckedout(bsa)
    return bsa.buffer[first(bsa.buffer_idx)]
end

# null functions: when using plain arrays
get_buffer(array::Union{SubArray, CuArray}) = array
function load_buffer!(array::Union{SubArray, CuArray})
    return nothing
end
function unload_buffer!(array::Union{SubArray, CuArray})
    return nothing
end
#end
