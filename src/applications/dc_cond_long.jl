## Special algorithm for longitudinal DC conductivity
function dc_long(
                 H, Jα,
                 H_rescale_factor,
                 NC::Int64, NR::Int64, NH::Int64;
                 verbose=0,
                 psi_in=nothing,
                 kernel=KPM.JacksonKernel,
                 Ef=0.0,
                 # workspace kwargs
                 ψr=maybe_on_device_zeros(dt_cplx, NH, NR * 2),
                 ψ0=maybe_on_device_zeros(dt_cplx, NH, NR * 2),
                 ψall_r=maybe_on_device_zeros(dt_cplx, NH, NR * 2, 3),
                )

    cond = on_host_zeros(dt_cplx, NC)

    kernel_vec = maybe_to_device(kernel.(0:NC-1, NC))
    kernel_vec .*= hn.(0:NC-1)

    Ef_tilde = Ef / H_rescale_factor

    Tn_e = chebyshevT_accurate.((1:NC) .- 1, Ef_tilde)


    if isnothing(psi_in)
        psi_in = exp.(rand(Float64, H_norm.n, NR) * 2im * pi);
        normalize_by_col(psi_in, NR)
    end
    psi_in = maybe_to_device(psi_in)

    H = maybe_to_device(H)
    Jα = maybe_to_device(Jα)

    # generate all views
    ψall_r_views = map(x -> view(ψall_r, :, :, x), 1:3)

    # right start
    view(ψ0, :, 1:NR) .= psi_in
    mul!(view(ψ0, :, (NR+1):(2*NR)), Jα, psi_in)

    # loop over r
    n = 1 # THIS IS g0, T0, etc.
    ψall_r_views[r_i(n)] .= ψ0
    ψr .+= ψall_r_views[r_i(n)] .* kernel_vec[n] .* Tn_e[n]

    n = 2
    mul!(ψall_r_views[r_i(n)], H, ψall_r_views[r_ip(n)])
    ψr .+= ψall_r_views[r_i(n)] .* kernel_vec[n] .* Tn_e[n]

    n_enum = 3:NC
    if verbose >= 1
        println("loop over n=3:$(NC)")
        n_enum = ProgressBar(n_enum)
    end
    for n in n_enum # TODO : save memory possible here. We do not need 3 vectors for psi 2
        chebyshev_iter_single(H,
                              ψall_r_views[r_ipp(n)],
                              ψall_r_views[r_ip(n)],
                              ψall_r_views[r_i(n)])
        ψr .+= ψall_r_views[r_i(n)] .* kernel_vec[n] .* Tn_e[n]

        cond[n] = dot(view(ψr, :, 1:NR), Jα, view(ψr, :, (NR+1):(NR*2)))
    end

    return cond / H_rescale_factor / NR
end


