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
                 ψall_r=maybe_on_device_zeros(dt_cplx, NH, NR * 2, 2),
                 avg_NR=true
                )
    Ef = KPM.dt_real(Ef)
    H_rescale_factor = KPM.dt_real(H_rescale_factor)
    #TODO assert NC_all is sorted
    NC_orig = NC_all
    NC_sort_i = sortperm(NC_orig, rev=true)
    NC_all = NC_orig[NC_sort_i]
    @assert issorted(NC_all, rev=true) "NC_all should be descend sorted"


    if !(typeof(kernel) <: Array)
        kernel = [kernel, kernel]
    end
    NC_max = maximum(NC_all)
    Ef_tilde = Ef / H_rescale_factor

    if Ef_tilde == 0
        Tn_e = chebyshevT_0.((0:NC_max-1)')
    else
        Tn_e = chebyshevT_accurate.((0:NC_max-1)', Ef_tilde)
    end

    kernel1_Tn = kernel[1].((0:NC_max-1)', NC_all) .* hn.((0:NC_max-1)') .* Tn_e
    kernel2_Tn = kernel[2].((0:NC_max-1)', NC_all) .* hn.((0:NC_max-1)') .* Tn_e

    kernel_Tn = maybe_to_device([kernel1_Tn kernel2_Tn])


    if isnothing(psi_in)
        psi_in = exp.(maybe_on_device_rand(dt_real, size(H, 1), NR) * 2im * pi);
        normalize_by_col(psi_in, NR)
    end
    psi_in = maybe_to_device(psi_in)

    H = maybe_to_device(H)
    Jα = maybe_to_device(Jα)

    # generate all views
    ψall_r_views = map(x -> view(ψall_r, :, :, x), 1:2)
    ψr_views = map(x -> view(ψr, :, :, x), 1:length(NC_all))

    # right start
    view(ψ0, :, 1:NR) .= psi_in
    @debug "$(size(psi_in)), $(size(Jα)), $(size(ψ0))"
    @sync mul!(view(ψ0, :, (NR+1):(2*NR)), Jα, psi_in)

    # loop over r
    n = 1 # THIS IS g0, T0, etc.
    @sync ψall_r_views[r2_i(n)] .= ψ0
    #NC_idx = findall(i -> i >= n, NC_all)
    NC_idx_max = findlast(i -> i >= n, NC_all)
    broadcast_assign!(ψr, ψr_views, ψall_r_views[r2_i(n)], kernel_Tn[:, n], NC_idx_max)

    n = 2
    @sync mul!(ψall_r_views[r2_i(n)], H, ψall_r_views[r2_ip(n)])
    #NC_idx = findall(i -> i >= n, NC_all)
    NC_idx_max = findlast(i -> i >= n, NC_all)
    @sync broadcast_assign!(ψr, ψr_views, ψall_r_views[r2_i(n)], kernel_Tn[:, n], NC_idx_max)

    n_enum = 3:NC_max
    if verbose >= 1
        println("loop over n=3:$(NC_max)")
        n_enum = ProgressBar(n_enum)
    end
    @sync begin
        for n in n_enum # TODO : save memory possible here. We do not need 3 vectors for psi 2
            @sync chebyshev_iter_single(H,
                                        ψall_r_views[r2_i(n)],
                                        ψall_r_views[r2_ip(n)])
            # output is stored at r2_i(n) === r2_ipp(n)

            #NC_idx = findall(i -> i >= n, NC_all)
            NC_idx_max = findlast(i -> i >= n, NC_all)
            @sync broadcast_assign!(ψr, ψr_views, ψall_r_views[r2_i(n)], kernel_Tn[:, n], NC_idx_max)
        end
    end

@time begin    
    ψr_views_1 = map(x -> view(ψr, :, 1:NR, x), 1:length(NC_all))
    ψr_views_2 = map(x -> view(ψr, :, (NR+1):(2*NR), x), 1:length(NC_all))

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
                @sync cond[NC_orig_i, NRi] = dot(view(ψr_views_1[NCi], :, NRi:NRi), Jα,  view(ψr_views_2[NCi], :, NRi:NRi))
            end
        end
    end
end

    return cond / H_rescale_factor
end


function broadcast_assign!(y_all::CuArray, y_all_views, x::CuArray, c_all::CuArray, idx_max::Int)
    # only working on 1:idx_max of NC_all
    block_count_x = cld(cld(length(x), 32), 512)
    block_count_y = idx_max
    @debug "block_count=$(block_count_x),$(block_count_y); c_all=$(c_all); idx=1:$(idx_max)"
    NVTX.@range "mainrange" begin
        CUDA.@sync @cuda threads=512 blocks=(block_count_x, block_count_y) cu_broadcast_assign!(y_all, x, c_all)
    end
    return nothing
end
function broadcast_assign!(y_all::Array, y_all_views::Array{T, 1} where {T<:Union{Array, SubArray}}, x::Union{Array, SubArray}, c_all::Array, idx_max::Int)
    if idx_max > Threads.nthreads()
        mt_broadcast_assign!(y_all_views, x, c_all, 1:idx_max)
    else
        finer_mt_broadcast_assign!(y_all_views, x, c_all, 1:idx_max)
    end
end


function cu_broadcast_assign!(y_all, x, c_all)
    # copying x to y_all (3D list, first two), multiplying by kernel_vecs_Tn
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x # thread id
    stride = blockDim().x * gridDim().x # number of threads per block * number of blocks
    c_idx = blockIdx().y

    x_l = length(x)
    for i = index:stride:x_l
        @inbounds y_all[i + (c_idx - 1) * x_l] += x[i] * c_all[c_idx]
    end
    return nothing
end

function mt_broadcast_assign!(y_all, x, c_all, idx)
    # copying x to y_all (list of list), multiplying by kernel_vecs_Tn
    Threads.@threads for j in idx
        @inbounds y_all[j] .+= x .* c_all[j]
    end
    return nothing
end

function finer_mt_broadcast_assign!(y_all, x, c_all, idx)
    # copying x to y_all (list of list), multiplying by kernel_vecs_Tn
    x_cols = size(x, 2)
    split_by_col = x_cols >= Threads.nthreads()
    if split_by_col
        N_splits = x_cols
        xv_all = map(i->view(x, :, i), 1:x_cols)
    else
        N_splits = Threads.nthreads()
        xv_all = _split_vector(x, N_splits)
    end

    for j in idx
        if split_by_col
            yjv_all = map(i->view(y_all[j], :, i), 1:x_cols)
        else
            yjv_all = _split_vector(y_all[j], N_splits)
        end
        cj = c_all[j]
        Threads.@threads for i in 1:length(xv_all)
            @inbounds yjv_all[i] .+= xv_all[i] .* cj
        end
    end

    return nothing
end

function _split_vector(x, N)
    pieces = _partition_l(length(x), N)

    ub = cumsum(pieces)
    lb = ub - pieces
    lb .+= 1

    return map((l,u)->view(x, l:u), lb, ub)
    
end

function _partition_l(l, N)
    each = cld(l, N) - 1
    res = fill(each, N)
    excess = l - each*N
    res[1:excess].+=1;
    return res
end
