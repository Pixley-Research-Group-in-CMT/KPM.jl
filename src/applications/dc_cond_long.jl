using Logging
## Special algorithm for longitudinal DC conductivity
function dc_long(
                 H, Jα,
                 H_rescale_factor,
                 NC_all::Vector{Int64}, NR::Int64, NH::Int64;
                 verbose=0,
                 psi_in=nothing,
                 kernel=KPM.JacksonKernel,
                 Ef=0.0,
                 # workspace kwargs
                 ψr=maybe_on_device_zeros(dt_cplx, NH, NR * 2, length(NC_all)),
                 ψ0=maybe_on_device_zeros(dt_cplx, NH, NR * 2),
                 ψall_r=maybe_on_device_zeros(dt_cplx, NH, NR * 2, 3),
                 avg_NR=true
                )

    #TODO assert NC_all is sorted
    NC_orig = NC_all
    NC_sort_i = sortperm(NC_orig, rev=true)
    NC_all = NC_orig[NC_sort_i]
    @assert issorted(NC_all, rev=true) "NC_all should be descend sorted"


    if !(typeof(kernel) <: Array)
        kernel = [kernel, kernel]
    end
    #kernel_vecs1 = map(NC -> kernel[1].(0:NC-1, NC) .* hn.(0:NC-1), NC_all)
    #kernel_vecs2 = map(NC -> kernel[2].(0:NC-1, NC) .* hn.(0:NC-1), NC_all)
    NC_max = maximum(NC_all)
    Ef_tilde = Ef / H_rescale_factor

    Tn_e = chebyshevT_accurate.((0:NC_max-1)', Ef_tilde)

    kernel1_Tn = kernel[1].((0:NC_max-1)', NC_all) .* hn.((0:NC_max-1)') .* Tn_e
    kernel2_Tn = kernel[2].((0:NC_max-1)', NC_all) .* hn.((0:NC_max-1)') .* Tn_e

    kernel_Tn = maybe_to_device([kernel1_Tn kernel2_Tn])


    if isnothing(psi_in)
        psi_in = exp.(rand(Float64, H.n, NR) * 2im * pi);
        normalize_by_col(psi_in, NR)
    end
    psi_in = maybe_to_device(psi_in)

    H = maybe_to_device(H)
    Jα = maybe_to_device(Jα)

    # generate all views
    ψall_r_views = map(x -> view(ψall_r, :, :, x), 1:3)
    ψall_r_views_1 = map(x -> view(ψall_r, :, 1:NR, x), 1:3)
    ψall_r_views_2 = map(x -> view(ψall_r, :, (NR+1):(2*NR), x), 1:3)
    ψr_views = map(x -> view(ψr, :, :, x), 1:length(NC_all))
    ψr_views_1 = map(x -> view(ψr, :, 1:NR, x), 1:length(NC_all))
    ψr_views_2 = map(x -> view(ψr, :, (NR+1):(2*NR), x), 1:length(NC_all))

    # right start
    view(ψ0, :, 1:NR) .= psi_in
    @debug "$(size(psi_in)), $(size(Jα)), $(size(ψ0))"
    @sync mul!(view(ψ0, :, (NR+1):(2*NR)), Jα, psi_in)

    # loop over r
    n = 1 # THIS IS g0, T0, etc.
    @sync ψall_r_views[r_i(n)] .= ψ0
    #NC_idx = findall(i -> i >= n, NC_all)
    NC_idx_max = findlast(i -> i >= n, NC_all)
    broadcast_assign!(ψr, ψall_r_views[r_i(n)], kernel_Tn[:, n], NC_idx_max)

    n = 2
    @sync mul!(ψall_r_views[r_i(n)], H, ψall_r_views[r_ip(n)])
    #NC_idx = findall(i -> i >= n, NC_all)
    NC_idx_max = findlast(i -> i >= n, NC_all)
    @time @sync broadcast_assign!(ψr, ψall_r_views[r_i(n)], kernel_Tn[:, n], NC_idx_max)

    n_enum = 3:NC_max
    if verbose >= 1
        println("loop over n=3:$(NC_max)")
        n_enum = ProgressBar(n_enum)
    end
    @sync begin
        for n in n_enum # TODO : save memory possible here. We do not need 3 vectors for psi 2
            @sync chebyshev_iter_single(H,
                                        ψall_r_views[r_ipp(n)],
                                        ψall_r_views[r_ip(n)],
                                        ψall_r_views[r_i(n)])


            #NC_idx = findall(i -> i >= n, NC_all)
            NC_idx_max = findlast(i -> i >= n, NC_all)
            if mod(n, 10) == 0
                @time broadcast_assign!(ψr, ψall_r_views[r_i(n)], kernel_Tn[:, n], NC_idx_max)
            else
                broadcast_assign!(ψr, ψall_r_views[r_i(n)], kernel_Tn[:, n], NC_idx_max)
            end
        end
    end

    @time begin
    if avg_NR
        cond = on_host_zeros(dt_cplx, length(NC_all))
        #Threads.@threads for NCi in 1:length(NC_all)
        for (NCi, NC_orig_i) in enumerate(NC_sort_i) #1:length(NC_all)
            @sync cond[NC_orig_i] = dot(ψr_views_1[NCi], Jα, ψr_views_2[NCi])
        end
        cond ./= NR
    else
        cond = on_host_zeros(dt_cplx, length(NC_all), NR)
        #Threads.@threads for NCi in 1:length(NC_all)
        for (NCi, NC_orig_i) in enumerate(NC_sort_i)
            for NRi in 1:NR
                @sync cond[NC_orig_i, NRi] = dot(view(ψr_views_1[NCi], :, NRi), Jα * view(ψr_views_2[NCi], :, NRi))
            end
        end
    end
end

    return cond / H_rescale_factor
end


function broadcast_assign!(y_all::CuArray, x::CuArray, c_all::CuArray, idx_max::Int)
    # only working on 1:idx_max of NC_all
    block_count_x = cld(cld(length(x), 8), 1024)
    block_count_y = idx_max
    @debug "block_count=$(block_count_x),$(block_count_y); c_all=$(c_all); idx=1:$(idx_max)"
    @cuda threads=1024 blocks=(block_count_x, block_count_y) cu_broadcast_mult_assign!(y_all, x, c_all)
    return nothing
end
function broadcast_assign!(y_all::Array{T, 1} where {T<:Union{Array, SubArray}}, x::Union{Array, SubArray}, c_all::Array, idx::Array)
    mt_broadcast_assign!(y_all, x, c_all, idx)
end


function cu_broadcast_mult_assign!(y_all, x, c_all)
    # copying x to y_all (3D list, first two), multiplying by kernel_vecs_Tn
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x # thread id
    stride = blockDim().x * gridDim().x # number of threads per block * number of blocks
    c_idx = blockIdx().y

    x_l = length(x)
    for i = index:stride:x_l
        #CUDA.@show "$i @ $(blockIdx().x)x$(threadIdx().x)"
        @inbounds y_all[i + (c_idx - 1) * x_l] += x[i] * c_all[c_idx]
    end
    return nothing
end
function cu_mult_assign!(y_all_j, x, c)
    # copying x to y_all (list of list), multiplying by kernel_vecs_Tn
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x # thread id
    stride = blockDim().x * gridDim().x # number of threads per block * number of blocks
    for i = index:stride:length(x)
        #CUDA.@show "$i @ $(blockIdx().x)x$(threadIdx().x)"
        y_all_j[i] += x[i] * c
    end
    return nothing
end

function mt_broadcast_assign!(y_all, x, c_all, idx)
    # copying x to y_all (list of list), multiplying by kernel_vecs_Tn
    Threads.@threads for j in idx
        y_all[j] .+= x .* c_all[j]
    end
    return nothing
end
