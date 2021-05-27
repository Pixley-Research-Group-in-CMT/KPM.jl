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
                )

    cond = on_host_zeros(dt_cplx, length(NC_all), NR)

    kernel_vecs = map(NC -> kernel.(0:NC-1, NC) .* hn.(0:NC-1), NC_all)

    Ef_tilde = Ef / H_rescale_factor

    NC_max = maximum(NC_all)
    Tn_e = chebyshevT_accurate.(0:NC_max-1, Ef_tilde)


    if isnothing(psi_in)
        psi_in = exp.(rand(Float64, H_norm.n, NR) * 2im * pi);
        normalize_by_col(psi_in, NR)
    end
    psi_in = maybe_to_device(psi_in)

    H = maybe_to_device(H)
    Jα = maybe_to_device(Jα)

    # generate all views
    ψall_r_views = map(x -> view(ψall_r, :, :, x), 1:3)
    ψr_views = map(x -> view(ψr, :, :, x), 1:length(NC_all))

    # right start
    view(ψ0, :, 1:NR) .= psi_in
    @debug "$(size(psi_in)), $(size(Jα)), $(size(ψ0))"
    mul!(view(ψ0, :, (NR+1):(2*NR)), Jα, psi_in)

    # loop over r
    n = 1 # THIS IS g0, T0, etc.
    ψall_r_views[r_i(n)] .= ψ0
    for NCi in 1:length(NC_all)
        ψr_views[NCi] .+= ψall_r_views[r_i(n)] .* kernel_vecs[NCi][n] .* Tn_e[n]
    end

    n = 2
    mul!(ψall_r_views[r_i(n)], H, ψall_r_views[r_ip(n)])
    for NCi in 1:length(NC_all)
        ψr_views[NCi] .+= ψall_r_views[r_i(n)] .* kernel_vecs[NCi][n] .* Tn_e[n]
    end

    n_enum = 3:NC_max
    if verbose >= 1
        println("loop over n=3:$(NC_max)")
        n_enum = ProgressBar(n_enum)
    end
    for n in n_enum # TODO : save memory possible here. We do not need 3 vectors for psi 2
        chebyshev_iter_single(H,
                              ψall_r_views[r_ipp(n)],
                              ψall_r_views[r_ip(n)],
                              ψall_r_views[r_i(n)])
        Threads.@threads for NCi in 1:length(NC_all)
            if n <= NC_all[NCi]
                ψr_views[NCi] .+= ψall_r_views[r_i(n)] .* kernel_vecs[NCi][n] .* Tn_e[n]
            end
        end
    end

    Threads.@threads for (NCi, NRi) in Iterators.product(1:length(NC_all), 1:NR)
        cond[NCi, NRi] = dot(view(ψr_views[NCi], :, NRi), Jα, view(ψr_views[NCi], :, NRi + NR))
    end

    return cond / H_rescale_factor
end


