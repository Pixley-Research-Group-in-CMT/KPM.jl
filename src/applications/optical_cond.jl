"""
- beta : Inf is zero temperature. beta = 1/T.

- E_f : Fermi energy. Between -1 and 1 because Hamiltonian is normalized. 
"""
function Λn(ntuple; E_f=0.0, beta=Inf, δ=1e-5, λ=0.0, quad=(f->quadgk(f, -1+δ, 1-δ)))
    # Equation 41, 
    # The integral will cover [-1+δ, 1-δ], where Fermi energy is taken care of by fermi function.
    # λ should be much smaller than δ to ensure the value of gn match with the λ->0
    # but usually not be too small to avoid floating point error dominated by large number near 1
    # future plan: if possible, try taking λ→0 analytically.
    #λ = δ / 100
    ff = fermiFunctions(E_f, beta)

    n = ntuple[1]
    # apply cutoffs
    _Δn(ϵ; n) = Δn(ϵ, n, δ)
    f_Δ(ϵ) = _Δn(ϵ; n=n)
    Λn_integrand(ϵ) = f_Δ(ϵ) * ff(ϵ)

    #I, E = quadgk(Λnmp_integrand, -1, E_f) # numerical integration, E is error
    I, E = quad(Λn_integrand)
    return I
end

function Λnm(nm, ω; E_f=0.0, beta=Inf, δ=1e-5, λ=0.0, quad=(f->quadgk(f, -1+δ, 1-δ)))
    # Equation 42, 
    # The integral will cover [-1+δ, 1-δ], where Fermi energy is taken care of by fermi function.
    # λ should be much smaller than δ to ensure the value of gn match with the λ->0
    # but usually not be too small to avoid floating point error dominated by large number near 1
    # future plan: if possible, try taking λ→0 analytically.
    #λ = δ / 100
    ff = fermiFunctions(E_f, beta)

    n, m = nm

    # apply cutoffs
    _gn_R(ϵ; n) = gn_R(ϵ, n, λ, δ) 
    _gn_A(ϵ; n) = gn_A(ϵ, n, λ, δ)
    _Δn(ϵ; n) = Δn(ϵ, n, δ)

    f_r(ϵ) = _gn_R(ϵ + ω; n=n) * _Δn(ϵ; n=m)
    f_a(ϵ) = _Δn(ϵ; n=n) * _gn_A(ϵ - ω; n=m)
    Λnm_integrand(ϵ) = (f_r(ϵ) + f_a(ϵ)) * ff(ϵ)

    #I, E = quadgk(Λnmp_integrand, -1, E_f) # numerical integration, E is error
    I, E = quad(Λnm_integrand)
    return I
end

function optical_cond1(Gamma, NC, ω; beta=Inf, E_f=0.0, kernel=JacksonKernel, δ=1e-5, Ω=ω/20)
    # Equation 44, last term
    # Gamma is calculated using Hamiltonian that is
    # normalized to have energy bounded by [-1, 1]
    #
    # Unit of -ie^2 / (ħ^2 * ω) is used.

    Gamma_tilde = muND_apply_kernel_and_h(Gamma, NC, kernel;dims=[1])

    # applying specified quad
    nodes, weights = gausschebyshev(NC * 8)
    quad(f) = (
               dot(weights, f.(nodes)),
               nothing # THIS SHOULD BE AN ESTIMATION OF ERROR
              )


    Λn_all = map(n -> Λn(n; δ=δ, E_f=E_f, beta=beta, quad=quad), Iterators.product(0:(NC-1)))

    Gamma_tilde .*= Λn_all
   
    return -1im * sum(Gamma_tilde) / ω
end

function d_optical_cond1(Gamma, NC; δ=1e-5, λ=0.0, kernel=JacksonKernel, N_int=NC*2, e_range=[-1.0, 1.0])
    
    ϵ_grid = collect((((0.5:N_int))/N_int * (e_range[2]-e_range[1]) .+ e_range[1])')

    Gamma = maybe_to_device(Gamma)

    res = d_optical_cond1.([Gamma], NC, ϵ_grid; δ=δ, λ=λ, kernel=kernel)
    return (ϵ_grid, res)
end

function d_optical_cond1(Gamma, NC, ϵ::Float64; δ=1e-5, λ=0.0, kernel=JacksonKernel,
               # pre-allocated arrays
               )
    Gamma = maybe_to_device(Gamma)
    
    n_grid = collect((0:(NC-1)))

    kernel_vec = kernel.(n_grid, NC)
    kernel_vec .*= hn.(n_grid)
    kernel_vec = maybe_to_device(kernel_vec)

    n_grid = convert(Vector{Float64}, n_grid)
    n_grid = maybe_to_device(n_grid)

    # each of the following have size (NC,)
    _Δn_ϵ = Δn.(ϵ, n_grid, δ)
    # indices n, m
    f_r = maybe_on_device_zeros(ComplexF64, NC)
    f_r .= reshape(kernel_vec, NC)
    f_r .*= reshape(_Δn_ϵ, NC)

    res = sum(f_r .* Gamma)
    return res
end


function optical_cond2(Gamma, NC, ω; beta=Inf, E_f=0.0, kernel=JacksonKernel, δ=1e-5, Ω=ω/20)
    # Equation 44, last term
    # Gamma is calculated using Hamiltonian that is
    # normalized to have energy bounded by [-1, 1]
    #
    # Unit of -ie^2 / (ħ^2 * ω) is used.

    Gamma_tilde = mu2D_apply_kernel_and_h(Gamma, NC, kernel)

    # applying specified quad
    nodes, weights = gausschebyshev(NC * 8)
    quad(f) = (
               dot(weights, f.(nodes)),
               nothing # THIS SHOULD BE AN ESTIMATION OF ERROR
              )


    Λnm_all = map(nm -> Λnm(nm, ω; δ=δ, E_f=E_f, beta=beta, quad=quad), Iterators.product(0:(NC-1), 0:(NC-1)))

    Gamma_tilde .*= Λnm_all
   
    return -1im * sum(Gamma_tilde) / ω
end



function d_optical_cond2(Gamma, NC, ω; δ=1e-5, λ=0.0, kernel=JacksonKernel, N_int=NC*2, e_range=[-1.0, 1.0])
    
    ϵ_grid = collect((((0.5:N_int))/N_int * (e_range[2]-e_range[1]) .+ e_range[1])')

    Gamma = maybe_to_device(Gamma)

    res = d_optical_cond2.([Gamma], NC, ω, ϵ_grid; δ=δ, λ=λ, kernel=kernel)
    return (ϵ_grid, res)
end

function d_optical_cond2(Gamma, NC, ω::Float64, ϵ::Float64; δ=1e-5, λ=0.0, kernel=JacksonKernel,
               # pre-allocated arrays
               )
    Gamma = maybe_to_device(Gamma)
    
    n_grid = collect((0:(NC-1)))

    kernel_vec = kernel.(n_grid, NC)
    kernel_vec .*= hn.(n_grid)
    kernel_vec = maybe_to_device(kernel_vec)

    n_grid = convert(Vector{Float64}, n_grid)
    n_grid = maybe_to_device(n_grid)

    # each of the following have size (NC,)
    _Δn_ϵ = Δn.(ϵ, n_grid, δ)

    # indices n, m
    f_r = maybe_on_device_zeros(ComplexF64, NC, NC)
    f_r .= reshape(kernel_vec, NC, 1)
    f_r .*= reshape(kernel_vec, 1, NC)
    f_a = copy(f_r)
    
    gn_ϵ_r = gn_R.(ϵ + ω, n_grid, λ, δ)
    f_r .*= reshape(gn_ϵ_r, NC, 1)

    f_r .*= reshape(_Δn_ϵ, 1, NC)

    f_a .*= reshape(_Δn_ϵ, NC, 1)

    gn_ϵ_a = gn_A.(ϵ .- ω, n_grid, λ, δ)
    f_a .*= reshape(gn_ϵ_a, 1, NC)
    
    res = sum((f_r + f_a) .* Gamma)
    return res
end
