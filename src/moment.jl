using DocStringExtensions
using ProgressBars

"""
$(TYPEDSIGNATURES)

Calculate the moments μ defined in KPM. 

- `H`           -- Hamiltonian. A matrix or sparse matrix

- `NC`          -- Integer. the cut off dimension

- `NR`          -- Integer. number of random vectors used for KPM evaluation

- `NH`          -- Integer. the size of hamiltonian

- `psi_in`      -- Optional. Allow setting random vector manually.

- `force_norm`  -- Boolean, Optional. Apply normalization.

- `verbose`     -- Integer. Default is 0. Enables progress bar if set `verbose=1`.

"""
function kpm_1d(
                H, NC::Int64, NR::Int64, NH::Int64;
                psi_in=nothing,
                psi_in_l=nothing,
                psi_in_r=nothing,
                force_norm=false,
                verbose=0
               )
    mu = on_host_zeros(dt_cplx, NC)
    if isnothing(psi_in)
        if (!isnothing(psi_in_l) | !isnothing(psi_in_r))
            @assert (!isnothing(psi_in_l) & !isnothing(psi_in_r)) "must set both `psi_in_l` and `psi_in_r` or neither."
            if force_norm
                normalize_by_col(psi_in_l, NR)
                normalize_by_col(psi_in_r, NR)
            end
            kpm_1d!(H, NC, NR, NH, mu, psi_in_l, psi_in_r)
        else
            kpm_1d!(H, NC, NR, NH, mu)
        end
    else
        @assert (isnothing(psi_in_l) & isnothing(psi_in_r)) "must either set `psi_in` or set `psi_in_l` and `psi_in_r`, but not both."
        if force_norm
            normalize_by_col(psi_in, NR)
        end
        kpm_1d!(H, NC, NR, NH, mu, psi_in)
    end

    return mu
end


"""
$(METHODLIST)

Calculate the moments μ defined in KPM. Output is saved in `mu`.

- `H`           -- Hamiltonian. A matrix or sparse matrix

- `NC`          -- Integer. the cut off dimension

- `NR`          -- Integer. number of random vectors used for KPM evaluation

- `NH`          -- Integer. the size of hamiltonian

- `mu`          -- Array. Output

- `psi_in`      -- Array (optional). Input array on the right side. A ket.


"""
function kpm_1d!(
                H, NC::Int64, NR::Int64, NH::Int64,
                mu,
                psi_in;
                verbose=0,
                # working arrays
                α_all = maybe_on_device_zeros(NH, NR, 2),
                μ_all = maybe_on_device_zeros(NR, NC)
                )
    H = maybe_to_device(H)

    @assert (mod(NC, 2) == 1) "Invalid NC: NC should be even."
    NChalf = div(NC,2)

    psi_in_size = size(psi_in)
    @assert (psi_in_size == (NH, NR)) "Invalid `psi_in` with size $(psi_in_size). Expecting $(NH), $(NR)"

    α_all[:, :, 1] = maybe_to_device(psi_in)
        
    mul!((@view α_all[:, :, 2]), H, (@view α_all[:, :, 1]))
    @. μ_all[:, 1] = 1.0
    mu1 = on_host_zeros(NR)

    # TODO this can be optimized
    for NRi in 1:NR
        mu1[NRi] = dot((@view α_all[:, NRi, 1]), (@view α_all[:, NRi, 2]))
    end
    mu1 = maybe_to_device(mu1)

    @. μ_all[:, 2] = mu1

    ip = 2
    ipp = 1

    n_enum = 2:NChalf
    if verbose >= 1
        n_enum = ProgressBar(n_enum)
    end

    α_views = [view(α_all, :, :, 1), view(α_all, :, :, 2)]
    for n=n_enum
        chebyshev_iter_single(H, α_views[ipp], α_views[ip])

        broadcast_dot_1d_1d!((@view μ_all[:, 2n-1]),
                             α_view[ip],
                             α_view[ip], NR, 2.0, -1.0)

        broadcast_dot_1d_1d!((@view μ_all[:, 2n]),
                             α_view[ip],
                             α_view[ipp],
                             NR, 2.0, -mu1)

        ip = 3-ip
        ipp = 3-ipp
    end

    mu .= maybe_to_host(real.(
                              dropdims(sum(μ_all, dims=1),
                                       dims=1)./NR
                             )
                       )
    return nothing
end
function kpm_1d!(
                H, NC::Int64, NR::Int64, NH::Int64,
                mu;
                verbose=0
                )
    psi_in = exp.(maybe_on_device_rand(NH, NR) * (2.0im * pi))
    normalize_by_col(psi_in, NR)
    kpm_1d!(H, NC, NR, NH, mu, psi_in, psi_in)
end
function kpm_1d!(
                H, NC::Int64, NR::Int64, NH::Int64,
                mu,
                psi_in_l, psi_in_r;
                verbose=0
                )
    # with different left and right.
    throw("unimplemented.")
end




"""
$(TYPEDSIGNATURES)

Calculate moments for 2D KPM. 

Calculates `ψ0l * Tm(H) * Jβ * Tn(H) * Jα * ψ0r`.
When ψ0r and ψ0l are chosen to be random and identical, the output approximates
tr(Tm(H) Jβ Tn(H) Jα). The accuracy is ~ O(1/sqrt(NR * NH)). NC controls the
energy resolution of the result.

Output: μ, a 2D array in ComplexF64. μ[n, m] is the momentum for 2D KPM.

**ARGS**

- `H`
  Hamiltonian. A sparse 2D array.

- `Jα`
  Current operator. A sparse 2D array.

- `Jβ`
  Current operator. A sparse 2D array.

- `NC`
  Integer. KPM cutoff order.

- `NR`
  Integer. Number of random vectors to choose from. When skipped, understood as NR=1.

- `NH`
  Integer. Dimension of H, Jα and Jβ

**KWARGS**

- `psi_in_l`

  Passes value to ψ0l. The array is not updated. Size should be
  (NH, NR) (preferred) or (NR, NH) if set.

- `psi_in_r`

  Passes value to ψ0r. The array is not updated. Size should be 
  (NH, NR) (preferred) or (NR, NH) if set.

- `psi_in`

  Cannot be used together with psi_in_l and psi_in_r. Sets psi_in_l=psi_in_r=psi_in if set.

- `kwargs`

  other kwargs in KPM_2D!

"""
function kpm_2d(
                H, Ja, Jb,
                NC::Int64, NR::Int64, NH::Int64;
                psi_in=nothing,
                psi_in_l=nothing,
                psi_in_r=nothing,
                kwargs...
               )
    throw("unimplemented")
end

"""
$(TYPEDSIGNATURES)

In place KPM2D. This is also the main building block for KPM_2D. This
method only provide NR=1.

Calculates `ψ0l * Tm(H) * Jβ * Tn(H) * Jα * ψ0r`.  When `ψ0r` and `ψ0l` are
chosen to be random and identical, the output approximates `tr(Tm(H) Jβ Tn(H) Jα)`.
The accuracy is ``\\sim O(1/sqrt(NR * NH))`` with NR repetitions. NC controls
the energy resolution of the result.

Output: nothing. Result is saved on μ.

**ARGS**

- `H` : Hamiltonian. A sparse 2D array.

- `Jα` : Current operator. A sparse 2D array.

- `Jβ` : Current operator. A sparse 2D array.

- `NC` : Integer. KPM cutoff order.

- `NR` : Integer. Number of random vectors.

- `NH` : Integer. Dimension of H, Jα and Jβ

- `μ` : 2D Array of dimension (NC, NC). Results will be updated here. Any data
will be overwritten.
    
- `psi_in` : Set `psi_in_l` and `psi_in_r`. Size is (NH, NR). The array is not updated.
Whether the input is normalized or not, it is assumed to be intended.
Usually `psi_in` should be normalized.

- `psi_in_l` : Passes value to ψ0l. Size is (NH, NR). The array is not updated.
Whether the input is normalized or not, it is assumed to be intended.
Usually `psi_in_l` should be normalized. `psi_in_l` is given as column vector
of ket ``|ψl> = <ψl|^\\dagger``

- `psi_in_r` : Passes value to ψ0r. Size is (NH, NR). The array is not updated.
Whether the input is normalized or not, it is assumed to be intended.
Usually `psi_in_r` should be normalized. `psi_in_r` is given as column vector
of ket ``|ψr>``. 

**KWARGS**

- `arr_size` : The buffer array size. Minimum is 3. Determines the number of
left states to be kept in memory for each loop of right states. The time
complexity is reduced from ``O(N\\times NC^2)`` to ``O(N\\times NC\\times arr\\_size)`` while space
complexity is increased from ``O(N\\times NC)`` to ``O(N\\times NC\\times arr\\_size)``.


**working spaces KWARGS**: The following keyword args are simply providing working
place arrays to avoid repetitive allocation and GC. They are automatically
created if not set. However, when using `KPM_2D!` for many times, it
is beneficial to reuse those arrays.  CONVENTION: args with `ψ` are all
working space arr.

- `ψ0r=maybe_on_device_zeros(NH, NR)`
- `Jψ0r=maybe_on_device_zeros(NH, NR)`
- `JTnHJψr=maybe_on_device_zeros(NH, NR)`
- `ψall_r=maybe_on_device_zeros(3, NH, NR)`
- `ψ0l=maybe_on_device_zeros(NH, NR)`
- `ψall_l=maybe_on_device_zeros(arr_size, NH, NR)`
- `ψw=maybe_on_device_zeros(NH, NR)`

"""
function kpm_2d!(
                 H, Jα, Jβ,
                 NC::Int64, NR::Int64, NH::Int64,
                 μ,
                 psi_in_l,
                 psi_in_r;
                 arr_size::Int64=3,
                 verbose=0,
                 # workspace kwargs
                 ψ0r=maybe_on_device_zeros(NH, NR),
                 Jψ0r=maybe_on_device_zeros(NH, NR),
                 JTnHJψr=maybe_on_device_zeros(NH, NR),
                 ψall_r=maybe_on_device_zeros(NH, NR, 3),
                 ψ0l=maybe_on_device_zeros(NH, NR),
                 ψall_l=maybe_on_device_zeros(NH, NR, arr_size),
                 ψw=maybe_on_device_zeros(NH, NR),
                )

    # do not enforce normalization
    @assert (size(psi_in_r) == (NH, NR)) "`psi_in_r` has size $(size(psi_in_r)) but expecting $(NH), $(NR)"
    @assert (size(psi_in_l) == (NH, NR)) "`psi_in_l` has size $(size(psi_in_l)) but expecting $(NH), $(NR)"
    ψ0r .= maybe_to_device(psi_in_r)
    ψ0l .= maybe_to_device(psi_in_l)

    H = maybe_to_device(H)
    Jα = maybe_to_device(Jα)
    Jβ = maybe_to_device(Jβ)

    # generate all views
    ψall_l_views = map(x -> view(ψall_l, :, :, x), 1:arr_size)
    ψall_r_views = map(x -> view(ψall_r, :, :, x), 1:3)

    # left starter
    ψall_l_views[1] .= ψ0l
    println("$(typeof(ψw)), $(typeof(H)), $(typeof(ψ0l))")
    mul!(ψw, H, ψ0l)
    ψall_l_views[2] .= ψw

    # right starter
    mul!(Jψ0r, Jα, ψ0r)

    reps = Integer(ceil(NC/arr_size - 1))

    for rep in 1:(reps+1)
        m1 = (rep - 1) * arr_size + 1
        m2 = min(rep * arr_size, NC)
        println("rep $(rep)/$(reps+1): $(m1) to $(m2)")
        rep_size = m2 - m1 + 1
        μ_rep = maybe_on_device_zeros(rep_size) # temp array for μ #Q: is there a better solution?
        # loop over l
        chebyshev_iter(H, ψall_l_views, rep_size)

        # loop over r
        n = 1
        ψall_r_views[n] .= Jψ0r
        mul!(JTnHJψr, Jβ,ψall_r_views[n])

        broadcast_dot_reduce_avg_2d_1d!(μ_rep, ψall_l_views, JTnHJψr, NR, rep_size)
        @. μ[n, m1:m2] = μ_rep
        ## TODO: IMPROVE THIS?

        # n = 2
        n = 2
        mul!(ψall_r_views[n], H, Jψ0r) # use initial values to calc Hψ0r
        mul!(JTnHJψr, Jβ, ψall_r_views[n])

        broadcast_dot_reduce_avg_2d_1d!(μ_rep, ψall_l_views, JTnHJψr, NR, rep_size)
        @. μ[n, m1:m2] = μ_rep

        n_enum = 3:NC
        if verbose >= 1
            n_enum = ProgressBar(n_enum)
        end
        for n in n_enum # TODO : save memory possible here. We do not need 3 vectors for psi 2
            chebyshev_iter_single(H,
                                  ψall_r_views[r_ipp(n)],
                                  ψall_r_views[r_ip(n)],
                                  ψall_r_views[r_i(n)])
            mul!(JTnHJψr, Jβ, ψall_r_views[r_i(n)])

            broadcast_dot_reduce_avg_2d_1d!(μ_rep, ψall_l_views, JTnHJψr, NR, rep_size)
            @. μ[n, m1:m2] = μ_rep
        end

        # wrap around to prepare for next
        chebyshev_iter_wrap(H, ψall_l_views, arr_size) #timed
    end

    return nothing
end
function kpm_2d!(
                 H, Jα, Jβ,
                 NC::Int64, NR::Int64, NH::Int64,
                 μ,
                 psi_in;
                 kwargs...
                )
    kpm_2d!(H, Jα, Jβ, NC, NR, NH, μ, psi_in, psi_in; kwargs...)
    return nothing
end
function kpm_2d!(
                 H, Jα, Jβ,
                 NC::Int64, NR::Int64, NH::Int64,
                 μ;
                 kwargs...
                )

    # random vector
    psi_in = exp.(2im*pi*maybe_on_device_rand(NH, NR))
    normalize_by_col(psi_in, NR)

    kpm_2d!(H, Jα, Jβ, NC, NR, NH, μ, psi_in; kwargs...)
    return nothing
end
