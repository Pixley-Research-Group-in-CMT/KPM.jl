## Special algorithm for longitudinal DC conductivity
function dc_long(
                 H, Jα,
                 H_rescale_factor,
                 NC::Int64, NR::Int64, NH::Int64;
                 verbose=0,
                 psi_in=exp.(maybe_on_device_rand(dt_real, NH, NR) * (2.0im * pi)),
                 kernel=KPM.JacksonKernel,
                 Ef=0.0,
                 # workspace kwargs
                 ψr=maybe_on_device_zeros(dt_cplx, NH, NR),
                 ψ0=maybe_on_device_zeros(dt_cplx, NH, NR),
                 ψall_r=maybe_on_device_zeros(dt_cplx, NH, NR, 3),
                )
    kernel_vec = maybe_to_device(kernel.(0:NC-1, NC))
    kernel_vec .*= hn.(0:NC-1)

    Ef_tilde = Ef / H_rescale_factor

    Tn_e = chebyshevT_accurate.((1:NC) .- 1, Ef_tilde)


    ψ0 .= maybe_to_device(psi_in)

    H = maybe_to_device(H)
    Jα = maybe_to_device(Jα)

    # generate all views
    ψall_r_views = map(x -> view(ψall_r, :, :, x), 1:3)

    # right start

    # loop over r
    n = 1 # THIS IS g0, T0, etc.
    mul!(ψall_r_views[r_i(n)], Jα, ψ0)
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
    end


    # loop over l
    m = 1 # g0, T0, etc.
    mul!(ψall_r_views[r_i(m)], Jα, ψr)
    ψr .= 0
    ψr .+= ψall_r_views[r_i(m)] .* kernel_vec[m] .* Tn_e[m]

    m = 2
    mul!(ψall_r_views[r_i(m)], H, ψall_r_views[r_ip(m)])
    ψr .+= ψall_r_views[r_i(m)] .* kernel_vec[m] .* Tn_e[m]

    m_enum = 3:NC
    if verbose >= 1
        println("loop over m=3:$(NC)")
        m_enum = ProgressBar(m_enum)
    end

    for m in m_enum # TODO : save memory possible here. We do not need 3 vectors for psi 2
        chebyshev_iter_single(H,
                              ψall_r_views[r_ipp(m)],
                              ψall_r_views[r_ip(m)],
                              ψall_r_views[r_i(m)])
        ψr .+= ψall_r_views[r_i(m)] .* kernel_vec[m] .* Tn_e[m]
    end

    return dot(ψr, ψ0) / H_rescale_factor
end
