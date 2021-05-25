using QuadGK # numerical integral
using FastGaussQuadrature # numerical integral - non adaptive
using Logging

function cpge(Gamma, NC, ω; beta=Inf, E_f=0.0, kernel=JacksonKernel, δ=1e-5, Ω=ω/20)
    # Equation 45, last term
    # Gamma is calculated using Hamiltonian that is
    # normalized to have energy bounded by [-1, 1]
    #
    # Unit of e^3/Ωħ^3

    @assert (abs.(Ω/ω) < 0.1) "Ω should be much smaller than ω."
    cpge_αβγ = 0.0 + 0im

    Gamma_tilde = mu3D_apply_kernel_and_h(Gamma, NC, kernel)

    # applying specified quad
    nodes, weights = gausschebyshev(NC * 8)
    quad(f) = (
               dot(weights, f.(nodes)),
               nothing # THIS SHOULD BE AN ESTIMATION OF ERROR
              )


    ω₁ = ω
    ω₂ = Ω - ω
    Λnmp_all = map(nmp -> Λnmp(nmp, ω₁, ω₂; δ=δ, E_f=E_f, beta=beta, quad=quad), Iterators.product(0:(NC-1), 0:(NC-1), 0:(NC-1)))

    Gamma_tilde .*= Λnmp_all
    #for n in 1:NC
    #    for m in 1:NC
    #        for p in 1:NC
    #            # note: this loop will be parallized
    #            Gamma_tilde[n, m, p] *= Λnmp([n-1, m-1, p-1], ω; δ=δ, E_f=E_f, beta=beta)
    #        end
    #    end
    #end
   
    return sum(Gamma_tilde) * 1im / (ω₁ * ω₂) * Ω
end

"""
- beta : Inf is zero temperature. beta = 1/T.

- E_f : Fermi energy. Between -1 and 1 because Hamiltonian is normalized. 
"""
function Λnmp(nmp, ω₁, ω₂; E_f=0.0, beta=Inf, δ=1e-5, λ=0.0, quad=(f->quadgk(f, -1+δ, 1-δ)))
    # Equation 43, Ω = ω1 + ω2, expecting Ω -> 0
    # The integral will cover [-1+δ, 1-δ], where Fermi energy is taken care of by fermi function.
    # λ should be much smaller than δ to ensure the value of gn match with the λ->0
    # but usually not be too small to avoid floating point error dominated by large number near 1
    # future plan: if possible, try taking λ→0 analytically.
    #λ = δ / 100
    ff = fermiFunctions(E_f, beta)

    n, m, p = nmp

    # apply cutoffs
    _gn_R(ϵ; n) = gn_R(ϵ, n, λ, δ) 
    _gn_A(ϵ; n) = gn_A(ϵ, n, λ, δ)
    _Δn(ϵ; n) = Δn(ϵ, n, δ)

    f_rr(ϵ) = _gn_R(ϵ + ω₁ + ω₂; n=n) * _gn_R(ϵ + ω₂; n=m) * _Δn(ϵ; n=p)
    f_ar(ϵ) = _gn_R(ϵ + ω₁; n=n) * _Δn(ϵ; n=m) * _gn_A(ϵ - ω₂; n=p)
    f_aa(ϵ) = _Δn(ϵ; n=n) * _gn_A(ϵ - ω₁; n=m) * _gn_A(ϵ - ω₁ - ω₂; n=p)
    Λnmp_integrand(ϵ) = (f_rr(ϵ) + f_ar(ϵ) + f_aa(ϵ)) * ff(ϵ)

    #I, E = quadgk(Λnmp_integrand, -1, E_f) # numerical integration, E is error
    I, E = quad(Λnmp_integrand)
    return I
end



function d_cpge(Gamma, NC, ω₁, ω₂; E_f=0.0, beta=Inf, δ=1e-5, λ=0.0, kernel=JacksonKernel, N_int=NC*2, e_range=[-1.0, 1.0])
    ϵ_grid = collect((((0.5:N_int))/N_int * (e_range[2]-e_range[1]) .+ e_range[1])')

    ff = fermiFunctions(E_f, beta)
    _ff_ϵ = ff.(ϵ_grid)
    @debug "$(ϵ_grid)"

    Gamma = maybe_to_device(Gamma)

    res = d_cpge.([Gamma], NC, ω₁, ω₂, ϵ_grid; δ=δ, λ=λ, kernel=kernel)
    return (ϵ_grid, res)
end
function d_cpge(Gamma, NC, ω₁::Float64, ω₂::Float64, ϵ::Float64; δ=1e-5, λ=0.0, kernel=JacksonKernel,
               # pre-allocated arrays
               )
    Gamma = maybe_to_device(Gamma)

    @debug "calculating for ϵ=$(ϵ)"
    f_rr = maybe_on_device_zeros(ComplexF64, NC, NC, NC)

    n_grid = maybe_to_device(collect((0:(NC-1))))

    # each of the following have size (NC,)
    _Δn_ϵ = Δn.(ϵ, n_grid, δ)
    @debug "size of _Δn_ϵ is $(size(_Δn_ϵ)), expecting $(NC)"

   
    kernel_vec = kernel.(n_grid, NC)
    kernel_vec .*= hn.(n_grid)
    @debug "size of kernel_vec is $(size(kernel_vec)), expecting $(NC)"

    # indices n, m, p
    f_rr .= reshape(kernel_vec, NC, 1, 1)
    f_rr .*= reshape(kernel_vec, 1, NC, 1)
    f_rr .*= reshape(kernel_vec, 1, 1, NC)
    f_ar = copy(f_rr)
    f_aa = copy(f_rr)
    
    gn_ϵ = gn_R.(ϵ + ω₁ + ω₂, n_grid, λ, δ)
    f_rr .*= reshape(gn_ϵ, NC, 1, 1)

    gn_ϵ = gn_R.(ϵ .+ ω₂, n_grid, λ, δ)
    f_rr .*= reshape(gn_ϵ, 1, NC, 1)

    f_rr .*= reshape(_Δn_ϵ, 1, 1, NC)

    gn_ϵ = gn_R.(ϵ .+ ω₁, n_grid, λ, δ)
    f_ar .*= reshape(gn_ϵ, NC, 1, 1)

    f_ar .*= reshape(_Δn_ϵ, 1, NC, 1)

    gn_ϵ = gn_A.(ϵ .- ω₂, n_grid, λ, δ)
    f_ar .*= reshape(gn_ϵ, 1, 1, NC)

    f_aa .*= reshape(_Δn_ϵ, NC, 1, 1)

    gn_ϵ = gn_A.(ϵ .- ω₁, n_grid, λ, δ)
    f_aa .*= reshape(gn_ϵ, 1, NC, 1)

    gn_ϵ = gn_A.(ϵ .- ω₁ .- ω₂, n_grid, λ, δ)
    f_aa .*= reshape(gn_ϵ, 1, 1, NC)

    res = sum((f_rr + f_ar + f_aa) .* Gamma)
    return res
end


function gn_A(ϵ, n, λ=0.0, δ=1e-5)
    # Equation 36 ctrl+k j3
    # λ is soft cutoff, δ is hard cutoff
    if abs(ϵ) > 1-δ
        return ϵ * 0
    end
    numerator = 2im * exp(1im * n * acos(ϵ - λ * im)) 
    denominator = sqrt(1 - (ϵ - λ * im)^2)
    return numerator / denominator
end

function gn_R(ϵ, n, λ=0.0, δ=1e-5)
    # Equation 36 ctrl+k j3
    # λ is soft cutoff, δ is hard cutoff
    if abs(ϵ) > 1-δ
        return ϵ * 0
    end
    numerator = - 2im * exp(- 1im * n * acos(ϵ + λ * im))
    denominator = sqrt(1 - (ϵ + λ * im)^2)
    return numerator / denominator
end

function Δn(ϵ, n, δ=1e-5)
    # Equation 35 ctrl+k D*
    if abs(ϵ) > 1-δ
        return ϵ * 0
    end
    numerator = 2 * cos(n * acos(ϵ))
    denominator = pi * sqrt(1 - ϵ^2)
    return numerator / denominator
end


