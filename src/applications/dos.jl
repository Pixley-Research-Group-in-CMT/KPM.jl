using DocStringExtensions
using ProgressBars

"""
$(METHODLIST)

Calculate DOS for a fermi energy grid spanning `E_range` with `N_tilde` total points.
If `E_range` is not set,  automatically set it to be sightly smaller than full size.
Otherwise an explicit array of `E_grid` can be passed in. Don't do both.

Either a) pass in a 1d array as moment and  as normalization factor; 
or b) pass in a Hamiltonian that is rescaled with an optional keyword
`rescale_factor` that default to 1.

- `H_rescale_factor` is the normalization of H. Needed when μ is passed.

- `NR` random vectors. Needed when H is passed

"""
function dos end

"""
$(METHODLIST)

Calculate DOS and energy derivatives at zero energy.
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
             fix_normalization = 0
            )
    H_rescale_factor, H_norm = normalizeH(H; fixed_a=fix_normalization)
    μ = kpm_1d(H, NC, NR)

    return dos(μ, H_rescale_factor; E_grid=E_grid, E_range=E_range, N_tilde=N_tilde, kernel=kernel, NC=NC)
end


function dos(
             μ, H_rescale_factor;
             E_grid=nothing,
             N_tilde::Int64=0,
             E_range=nothing,
             NC::Int64=0,
             kernel = JacksonKernel,
            )
    @assert (length(size(μ)) == 1) "The input need to be 1D array"
    μ = maybe_to_device(μ)
    @assert H_rescale_factor > 0
    a = H_rescale_factor # for convenience

    if (NC==0)
        NC = length(μ)
    else
        NC = min(NC,length(μ))
    end


    if isnothing(E_grid)
        if N_tilde == 0
            N_tilde = NC * 2
            println("Setting N_tilde=$(N_tilde).")
        end
        if isnothing(E_range) 
            E_range = [-a+0.01, a-0.01]
            println("Setting E_range=$(E_range).")
        end
        E_grid = collect(((0:(N_tilde)).*(E_range[2]-E_range[1]))./N_tilde .+ E_range[1])

    else
        @assert isnothing(E_range) "Should not set `E_grid` and `E_range` simoutaneously."
        @assert (length(E_grid) == N_tilde) """`N_tilde` does not match with `E_grid`.
        `N_tilde` is only necessary when using `E_range` instead of `E_grid`"""
    end



    rhoE_full = similar(E_grid)
    rhoE_full .= 0


    idx = (abs.(E_grid) .< abs(a))
    E_grid_inrange = maybe_to_device(E_grid[idx])

    n_grid = maybe_to_device(collect(0:(NC-1)))

    hgn = maybe_to_device(kernel.(0:(NC-1),NC) .* hn.(0:(NC-1)))

    rhoE = chebyshev_lin_trans(E_grid_inrange / a,
                               n_grid,
                               μ .* hgn)

    denom = @. (a*pi*sqrt(1-(E_grid_inrange/a)^2))
    rhoE ./= denom

    rhoE_full[idx] = maybe_to_host(rhoE)
    return E_grid, rhoE_full
end
