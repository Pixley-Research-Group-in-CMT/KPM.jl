using DocStringExtensions
using ProgressBars
using Zygote

"""
$(METHODLIST)

Calculate the DOS ρ(E) on an energy grid.

Either a) pass the moments `μ` (from [`kpm_1d`](@ref)) together with the
rescale factor `a = H_rescale_factor` (and, for a center-shifted rescaling,
the shift `b`); or b) pass the Hamiltonian `H` directly, which is rescaled
internally.

The reconstruction is
ρ(E) = Σₙ hₙ gₙ μₙ Tₙ(x) / (a π √(1-x²)),  x = (E - b)/a,
with gₙ the damping kernel (Jackson by default) and h₀=1, hₙ=2 for n ≥ 1.

- `E_grid` : explicit energies to evaluate; or set `E_range=[Emin, Emax]`
  with `N_tilde` points. Default covers the rescaled band with a 1% margin.
  Points outside the band return ρ = 0.

- `b` : center shift of the rescaling, `H_norm = (H - b I)/a`. Default 0.
  Use `normalizeH(H; center=true)` to obtain `(a, b, H_norm)`.

- `dE_order` : return the dE_order-th energy derivative instead (currently
  ≤ 1; use `dos0(μ, a; dE_order=2)` at the band center).

- `NR` : number of random vectors (only used when `H` is passed).
"""
function dos end

"""
$(METHODLIST)

DOS (and, via `dE_order=2`, its second energy derivative) at the band
center of the rescaling, i.e. at x = (E-b)/a = 0. With the default
symmetric rescaling (b = 0) this is the DOS at E = 0. Uses the closed-form
values of Tₙ(0), so only moments with even n contribute.
"""
function dos0 end


function dos(
             H;
             NC::Int64=1024,
             NR::Int64=12,
             E_grid=nothing,
             N_tilde::Int64=0,
             E_range=nothing,
             kernel = JacksonKernel,
             fix_normalization = 0,
             dE_order=0,
             center=false
            )
    if center
        a, b, H_norm = normalizeH(H; fixed_a=fix_normalization, center=true)
    else
        a, H_norm = normalizeH(H; fixed_a=fix_normalization)
        b = 0.0
    end
    μ = kpm_1d(H_norm, NC, NR)

    return dos(μ, a; b=b, E_grid=E_grid, E_range=E_range, N_tilde=N_tilde, kernel=kernel, NC=NC, dE_order=dE_order)
end


function dos(
             μ, H_rescale_factor;
             b=0.0,
             E_grid=nothing,
             N_tilde::Int64=0,
             E_range=nothing,
             NC::Int64=0,
             kernel=JacksonKernel,
             dE_order=0
            )
    @assert (length(size(μ)) == 1) "The input need to be 1D array"
    μ = maybe_to_device(μ)
    @assert H_rescale_factor > 0
    a = H_rescale_factor # for convenience

    if (NC == 0)
        NC = length(μ)
    else
        if NC > length(μ)
            @warn "NC=$(NC) exceeds the number of moments; decreased to $(length(μ))."
        end
        NC = min(NC, length(μ))
    end

    if isnothing(E_grid)
        if N_tilde == 0
            N_tilde = NC * 2
        end
        if isnothing(E_range)
            E_range = [b - 0.99 * a, b + 0.99 * a]
        end
        E_grid = collect(((0:(N_tilde)).*(E_range[2]-E_range[1]))./N_tilde .+ E_range[1])
    else
        @assert isnothing(E_range) "Should not set `E_grid` and `E_range` simultaneously."
        if N_tilde != 0
            @assert (length(E_grid) == N_tilde) """`N_tilde` does not match with `E_grid`.
            `N_tilde` is only necessary when using `E_range` instead of `E_grid`"""
        end
    end

    rhoE_full = similar(E_grid, float(eltype(E_grid)))
    rhoE_full .= 0

    # only x = (E-b)/a strictly inside (-1, 1) is reconstructable
    idx = (abs.(E_grid .- b) .< a)
    x_grid_inrange = maybe_to_device((E_grid[idx] .- b) ./ a)

    hgn = maybe_to_device(kernel.(0:(NC-1),NC) .* hn.(0:(NC-1)))
    μtilde = μ .* hgn

    if dE_order == 0
        n_grid = maybe_to_device(collect(0:(NC-1)), Int64)
        rhoE = chebyshev_lin_trans(x_grid_inrange, n_grid, μtilde)

        denom = @. (a * pi * sqrt(1 - x_grid_inrange^2))
        rhoE ./= denom
    else
        f(E) = _dos_single(μtilde, a, b, E, NC)
        g(E) = real(Zygote.forwarddiff(f, E))
        @assert (dE_order <= 1) "There is a Zygote support problem for higher order derivative. You can use `KPM.dos0(; dE_order=2)` for second derivative at the band center temporarily."
        for dE_order_i = 1:dE_order
            g = real ∘ g'
        end
        rhoE = g.(E_grid[idx])
    end

    rhoE_full[idx] = maybe_to_host(rhoE)
    return E_grid, rhoE_full
end


function _dos_single(μtilde, H_rescale_factor, b, E::Number, NC)
    ### MUST BE NON-MUTATING ###
    a = H_rescale_factor
    x = (E - b) / a
    n_grid = collect(0:(NC - 1))
    rhoE = chebyshev_lin_trans(x, n_grid, μtilde)

    denom = @. (a * pi * sqrt(1 - x^2))
    rhoE /= denom
    return rhoE
end

function dos0(
              μ, H_rescale_factor;
              NC::Int64=0,
              kernel=JacksonKernel,
              dE_order=0
             )
    @assert H_rescale_factor > 0
    a = H_rescale_factor # for convenience

    if NC == 0
        NC = length(μ)
    else
        if NC > length(μ)
            @warn "NC=$(NC) exceeds the number of moments; decreased to $(length(μ))."
        end
        NC = min(NC, length(μ))
    end

    # Tₙ(0) = +1 for n ≡ 0 (mod 4), -1 for n ≡ 2 (mod 4), 0 for odd n
    n_0 = 0:4:(NC-1)
    n_2 = 2:4:(NC-1)

    mu_n0 = μ[n_0 .+ 1]
    mu_n2 = μ[n_2 .+ 1]

    if dE_order == 0
        res = 0
        res += sum(@. mu_n0 * hn(n_0) * kernel(n_0, NC))
        res -= sum(@. mu_n2 * hn(n_2) * kernel(n_2, NC))
        res /= (a * pi)
        return res
    elseif dE_order == 2
        # d²/dx² [Tₙ(x)/√(1-x²)] at x=0 is (-1)^{n/2} (1-n²) for even n
        res = 0
        res += sum(@. mu_n0 * hn(n_0) * kernel(n_0, NC) * (1 - n_0 ^ 2))
        res += sum(@. mu_n2 * hn(n_2) * kernel(n_2, NC) * (n_2 ^ 2 - 1))
        res /= (a^3 * pi)
        return res
    else
        throw("unimplemented for derivative of order $(dE_order).")
    end

end
