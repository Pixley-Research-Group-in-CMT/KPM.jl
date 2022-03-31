function ldos(
                 H, NC::Int64, site::Int64;
                 kwargs...
                )
    NH = size(H)[1]
    mu = zeros(ComplexF64,1, NC) #on_host_zeros(dt_cplx, NR, NC)
    sitevector = zeros(NH,1)
    sitevector[site] = 1
    psi_in = sitevector #KPM.maybe_on_device(sitevector) 
    KPM.kpm_1d!(H, NC, 1, NH, mu, psi_in; kwargs...)
    mu = dropdims(mu, dims=1)
    mu = real.(mu)
    return mu
end
