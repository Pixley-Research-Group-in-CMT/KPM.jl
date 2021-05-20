using DocStringExtensions
using ProgressBars
using Zygote
using Logging 
include("dc_cond_util.jl")

"""
$(METHODLIST)

Calculate the integrand for conductivity of an energy grid spanning `E_range`
with `N_tilde` total points.  If `E_range` is not set,  automatically set it to
be sightly smaller than full size.  Otherwise an explicit array of `E_grid` can
be passed in. Don't do both.

Either a) pass in a 2d array as moment and a normalization factor; 
or b) pass in a Hamiltonian that is rescaled with an optional keyword
`rescale_factor` that default to 1, as well as two current operators Jα and Jβ

- `H_rescale_factor` is the normalization of H. Needed when μ is passed.

- `NR` random vectors. Needed when H is passed

"""
function d_dc_cond end

"""
$(METHODLIST)

Calculate DOS and its energy derivatives (by setting `dE_order`) at zero energy.

Warning: This method does not have correct normalization at the moment. 
"""
function dc_cond0 end

"""
$(METHODLIST)

Calculate DOS at a given energy.
"""
function dc_cond_single end




function d_dc_cond_old(μ, a::Float64; E_grid=nothing, NC::Integer=0, kernel=JacksonKernel)
    μ=complex(μ)
    μ=maybe_to_device(μ) # temporary

    if (NC==0)
        # if not specified, take full
        NC = size(μ)[1]
    else 
        # not allowing it to exceed actual NC for the  calculated μ.
        println("NC=$(NC) exceeds the maximum size of μ. Changing to $(size(μ)[1])")
        NC = min(size(μ)[1],NC)
    end

    if isnothing(E_grid)
        Erange = [-a+0.01,a-0.01]
        Ntilde = 2 * NC
        E_grid = collect(((0:(Ntilde)).*(Erange[2]-Erange[1]))./Ntilde .+ Erange[1])
    end

    dσE_full = similar(E_grid, ComplexF64)
    @. dσE_full *= 0
    idx = findall(abs.(E_grid) .< abs(a))

    #process μtilde
    μtilde = mu2D_apply_kernel_and_h(μ, NC, kernel)


    for idx_ in idx
        ϵ = E_grid[idx_] / a
        dσE_full[idx_] = Γnmμnmαβ(μtilde, ϵ, NC) / ((1-ϵ^2)^2) / (a^2)
    end

    return E_grid, dσE_full
end


d_dc_cond(μ, a::Float64, E::Float64; kwargs...) = d_dc_cond(μ, a, [E]; kwargs...)

function d_dc_cond(μ, a::Float64, E::Array{Float64, 1}; NC::Integer=0, kernel=JacksonKernel, dE_order=0)
    # TODO how to share some of the code with other methods?
    
    if (NC==0)
        # if not specified, take full
        NC = size(μ)[1]
    else 
        # not allowing it to exceed actual NC for the  calculated μ.
        println("NC=$(NC) exceeds the maximum size of μ. Changing to $(size(μ)[1])")
        NC = min(size(μ)[1],NC)
    end

    dσE = similar(E, Float64) * 0

    #process μtilde
    μtilde = mu2D_apply_kernel_and_h(μ, NC, kernel)
    
    f(x) = _d_dc_cond_single(μtilde, a, x, NC)
    g(x) = real(Zygote.forwarddiff(f, x))
    for dE_order_i = 1:dE_order
        g = real ∘ g'
    end

    idx = abs.(E) .< abs(a)
    dσE[idx] .= real(g.(E[idx]))
    return dσE

end


function _d_dc_cond_single(μtilde, H_rescale_factor::Float64, E, NC::Int64)
    a = H_rescale_factor

    ϵ = E / a
    dσE = real(Γnmμnmαβ(μtilde, ϵ, NC) / ((1-ϵ^2)^2) / (a^2))

    return dσE
end

"""
calculate conductivity for EF=0. Only works for longitudinal cond.
Garcia et. al. Supp eq.25.
"""
function dc_cond0(mu, H_rescale_factor::Number; kernel=JacksonKernel, NC::Int64=size(mu, 1))
    mu_tilde = mu2D_apply_kernel_and_h(mu[1:NC, 1:NC], NC, kernel)
    oneone = sum(mu_tilde[1:4:NC, 1:4:NC]);
    onethree = sum(mu_tilde[1:4:NC, 3:4:NC]);
    threeone = sum(mu_tilde[3:4:NC, 1:4:NC]);
    threethree = sum(mu_tilde[3:4:NC, 3:4:NC]);
    return (oneone + threethree - onethree - onethree) / H_rescale_factor # is it correct to divide by H_rescale_factor?
end
"""
calculate conductivity for arbitrary EF. Only works for longitudinal cond.
Garcia et. al. Supp eq.25.
"""
function dc_cond_single(mu, H_rescale_factor::Number, Ef::Number; kernel=JacksonKernel, NC::Int64=size(mu, 1))
    Ef_tilde = Ef / H_rescale_factor
    mu_tilde = maybe_to_host(mu2D_apply_kernel_and_h(mu[1:NC, 1:NC], NC, kernel))
    
    chebyT_poly_all = chebyshevT_accurate.((1:NC) .- 1, Ef_tilde)
    @debug "$(size(chebyT_poly_all)), $(size(mu_tilde))"

    mu_tilde .*= chebyT_poly_all
    mu_tilde .*= chebyT_poly_all'

    return sum(mu_tilde) / H_rescale_factor # is it correct to divide by H_rescale_factor?

end
