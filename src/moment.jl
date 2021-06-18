using DocStringExtensions
using ProgressBars
using SparseArrays
using Logging


"""
$(METHODLIST)

The in-place version of 1D KPM. 
Calculate the moments μ defined in KPM. Output is saved in `mu`.

- `H`           -- Hamiltonian. A matrix or sparse matrix.

- `NC`          -- Integer. the cut off dimension.

- `NR`          -- Integer. number of random vectors used for KPM evaluation.

- `NH`          -- Integer. the size of hamiltonian.

- `mu_all`          -- Array. Output for each random vector. Size (NR, NC). 

- `psi_in`      -- Array (optional). Input array on the right side. A ket.

"""
function kpm_1d! end

"""
$(METHODLIST)

The simple version of 1D KPM that returns the moment.

- `H`           -- Hamiltonian. A matrix or sparse matrix

- `NC`          -- Integer. the cut off dimension

- `NR`          -- Integer. number of random vectors used for KPM evaluation

- `NH`          -- Integer. the size of hamiltonian

- `psi_in`      -- Optional. Allow setting random vector manually.

- `force_norm`  -- Boolean, Optional. Apply normalization.

- `verbose`     -- Integer. Default is 0. Enables progress bar if set `verbose=1`.

- `avg_output`  -- Boolean. Default is true. Whether to output averaged μ (hence size NC) or separate μs (size NR x NC).

"""
function kpm_1d end

"""
$(METHODLIST)

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

- `moment_parity` : The condition enforced on μmn. Choose from `:NONE`, `:ODD` and `:EVEN`.
`:NONE` will calculate all μmn; `:ODD` will calculate μmn such that `mod(m+n, 2)==1`;
`:EVEN` will calculate μmn such that `mod(m+n, 2)==0`. As an example, `moment_parity=:EVEN`
can be used when calculating longitudinal conductivity on model with
particle-hole symmetry to save time and increase accuracy. 


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
function kpm_2d! end

"""
$(METHODLIST)

The simple version of 2D KPM that returns the moment.
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
function kpm_2d end



"""
$(METHODLIST)

The simple version of tripple KPM that returns the moment.
Calculate moments for tripple KPM. 

Calculates `ψ0l * Tn3(H) * Jγ * Tn2(H) * Jβ * Tn1(H) * Jα * ψ0r`.
When ψ0r and ψ0l are chosen to be random and identical, the output approximates
tr(Tn3(H) Jγ Tn2(H) Jβ Tn1(H) Jα). The accuracy is ~ O(1/sqrt(NR * NH)). NC controls the
energy resolution of the result.

Output: μ, a 3D array in ComplexF64. μ[n3, n2, n1] is the momentum for 2D KPM.

**ARGS**

- `H`
Hamiltonian. A sparse 2D array.

- `Jα`
Current operator. A sparse 2D array.

- `Jβ`
Current operator. A sparse 2D array.

- `Jγ`
Current operator. A sparse 2D array.

- `NC`
Integer. KPM cutoff order.

- `NR`
Integer. Number of random vectors to choose from. When skipped, understood as NR=1.

- `NH`
Integer. Dimension of H, Jα, Jβ and Jγ

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
function kpm_3d! end


"""
$(METHODLIST)

TODO: add doc.
"""
function kpm_3d end


kpm_1d(H, NC::Int64, NR::Int64; kwargs...) = kpm_1d(H, NC, NR, size(H)[1]; kwargs...)
function kpm_1d(
                H, NC::Int64, NR::Int64, NH::Int64;
                psi_in=nothing,
                psi_in_l=nothing,
                psi_in_r=nothing,
                force_norm=false,
                verbose=0,
                avg_output=true,
                NR_parallel=true
               )
    
  
    mu_all = on_host_zeros(dt_cplx, NR, NC) # this mu is never large enough to be worth putting on GPU
    if isnothing(psi_in)
        if (!isnothing(psi_in_l) | !isnothing(psi_in_r))
            @assert (!isnothing(psi_in_l) & !isnothing(psi_in_r)) "must set both `psi_in_l` and `psi_in_r` or neither."
            if force_norm
                normalize_by_col(psi_in_l, NR)
                normalize_by_col(psi_in_r, NR)
            end
            if NR_parallel
                kpm_1d!(H, NC, NR, NH, mu_all, psi_in_l, psi_in_r; verbose=verbose)
            else
                for NRi = 1:NR
                    kpm_1d!(H, NC, 1, NH, view(mu_all, NRi:NRi, :), view(psi_in_l, NRi:NRi, :), view(psi_in_r, NRi:NRi, :); verbose=verbose)
                end
            end
        else
            if NR_parallel
                kpm_1d!(H, NC, NR, NH, mu_all; verbose=verbose)
            else
                for NRi = 1:NR
                    kpm_1d!(H, NC, 1, NH, view(mu_all, NRi:NRi, :); verbose=verbose)
                end
            end
        end
    else
        @assert (isnothing(psi_in_l) & isnothing(psi_in_r)) "must either set `psi_in` or set `psi_in_l` and `psi_in_r`, but not both."
        if force_norm
            normalize_by_col(psi_in, NR)
        end
        if NR_parallel
            kpm_1d!(H, NC, NR, NH, mu_all, psi_in; verbose=verbose)
        else
            for NRi = 1:NR
                kpm_1d!(H, NC, 1, NH, view(mu_all, NRi:NRi, :), view(psi_in, NRi:NRi, :); verbose=verbose)
            end
        end
    end

    if avg_output
        return maybe_to_host(real.(
                                   dropdims(sum(mu_all, dims=1),
                                            dims=1)./NR
                                  )
                            )
    end

    return mu
end

function kpm_1d!(
                 H, NC::Int64, NR::Int64, NH::Int64,
                 mu_all,
                 psi_in;
                 verbose=0,
                 # working arrays
                 α_all = maybe_on_device_zeros(dt_cplx, NH, NR, 2),
                )
    @assert size(mu_all) == (NR, NC)
    H = maybe_to_device(H)

    @assert (mod(NC, 2) == 0) "Invalid NC: NC should be even."
    NChalf = div(NC,2)

    psi_in_size = size(psi_in)
    @assert (psi_in_size == (NH, NR)) "Invalid `psi_in` with size $(psi_in_size). Expecting $(NH), $(NR)"

    α_all[:, :, 1] = maybe_to_device(psi_in)

    mul!((@view α_all[:, :, 2]), H, (@view α_all[:, :, 1]))
    @. mu_all[:, 1] = 1.0
    mu1 = on_host_zeros(dt_cplx, NR)

    # TODO this can be optimized
    for NRi in 1:NR
        mu1[NRi] = dot((@view α_all[:, NRi, 1]), (@view α_all[:, NRi, 2]))
    end

    @. mu_all[:, 2] = mu1

    ip = 2
    ipp = 1

    n_enum = 2:NChalf
    if verbose >= 1
        println("NC/2 = $(NC/2)")
        n_enum = ProgressBar(n_enum)
    end

    α_views = [view(α_all, :, :, 1), view(α_all, :, :, 2)]
    split_views = x -> (map(i -> view(x, :, i), 1:NR))
    α_view_views = map(split_views, α_views) # array of CuArrays or array of SubArrays
    mu_all_views = map(i -> view(mu_all, :, i), 1:NC)
    for n=n_enum
        chebyshev_iter_single(H, α_views[ipp], α_views[ip])

        broadcast_dot_1d_1d!(mu_all_views[2n-1],
                             α_view_views[ip],
                             α_view_views[ip];
                             alpha=2.0, beta=-1.0)

        broadcast_dot_1d_1d!(mu_all_views[2n],
                             α_view_views[ip],
                             α_view_views[ipp];
                             alpha=2.0, beta=-mu1)

        ip = 3-ip
        ipp = 3-ipp
    end

    return nothing
end
function kpm_1d!(
                 H, NC::Int64, NR::Int64, NH::Int64,
                 mu;
                 kwargs...
                )
    psi_in = exp.(maybe_on_device_rand(dt_real, NH, NR) * (2.0im * pi))
    normalize_by_col(psi_in, NR)
    kpm_1d!(H, NC, NR, NH, mu, psi_in; kwargs...)
end
function kpm_1d!(
                 H, NC::Int64, NR::Int64, NH::Int64,
                 mu,
                 psi_in_l, psi_in_r;
                 kwargs...
                )
    # with different left and right.
    throw("unimplemented.")
end



function kpm_2d(
                H, Jα, Jβ,
                NC::Int64, NR::Int64, NH::Int64;
                psi_in=nothing,
                psi_in_l=nothing,
                psi_in_r=nothing,
                arr_size=3,
                moment_parity=:NONE,
                verbose=0
               )
    mu = on_host_zeros(dt_cplx, NC, NC)
    if isnothing(psi_in) & isnothing(psi_in_l) & isnothing(psi_in_r)
        kpm_2d!(H, Jα, Jβ, NC, NR, NH, mu; arr_size=arr_size, verbose=verbose, moment_parity=moment_parity)
    elseif !isnothing(psi_in) & isnothing(psi_in_l) & isnothing(psi_in_r)
        kpm_2d!(H, Jα, Jβ, NC, NR, NH, mu, psi_in; arr_size=arr_size, verbose=verbose, moment_parity=moment_parity)
    elseif isnothing(psi_in) & !isnothing(psi_in_l) & !isnothing(psi_in_r)
        kpm_2d!(H, Jα, Jβ, NC, NR, NH, mu, psi_in_l, psi_in_r; arr_size=arr_size, verbose=verbose, moment_parity=moment_parity)
    else
        throw("unimplemented")
    end
    return mu
end

function kpm_2d!(
                 H, Jα, Jβ,
                 NC::Int64, NR::Int64, NH::Int64,
                 μ,
                 psi_in_l,
                 psi_in_r;
                 arr_size::Int64=3,
                 verbose=0,
                 mn_sym=false,
                 moment_parity=:NONE,
                 # workspace kwargs
                 ψ0r=maybe_on_device_zeros(dt_cplx, NH, NR),
                 Jψ0r=maybe_on_device_zeros(dt_cplx, NH, NR),
                 JTnHJψr=maybe_on_device_zeros(dt_cplx, NH, NR),
                 ψall_r=maybe_on_device_zeros(dt_cplx, NH, NR, 3),
                 ψ0l=maybe_on_device_zeros(dt_cplx, NH, NR),
                 ψall_l=maybe_on_device_zeros(dt_cplx, NH, NR, arr_size),
                 ψw=maybe_on_device_zeros(dt_cplx, NH, NR),
                )

    if moment_parity == :NONE
        _NC_offset = 0
        NCstep = 1
    elseif moment_parity == :ODD # odd means we only consider terms that mod(m-n, 2)==1 (even-odd or odd-even)
        _NC_offset = 1
        NCstep = 2
    elseif moment_parity == :EVEN # even means we only consider terms that mod(m-n, 2)==0 (even-even or odd-odd)
        _NC_offset = 0
        NCstep = 2
    else
        throw(ArgumentError("moment_parity=$(moment_parity) not understood."))
    end
    NC0(m1, n) = mod(m1 + n + _NC_offset, NCstep) + 1

    # do not enforce normalization
    @assert (size(psi_in_r) == (NH, NR)) "`psi_in_r` has size $(size(psi_in_r)) but expecting $(NH), $(NR)"
    @assert (size(psi_in_l) == (NH, NR)) "`psi_in_l` has size $(size(psi_in_l)) but expecting $(NH), $(NR)"
    ψ0r .= maybe_to_device(psi_in_r)
    ψ0l .= maybe_to_device(psi_in_l)

    #mn_sym = false
    #if Jα ≡ Jβ
    #    mn_sym = true
    #    println("Jα and Jβ are identical. using m <-> n symmetry.")
    #end

    H = maybe_to_device(H)
    Jα = maybe_to_device(Jα)
    Jβ = maybe_to_device(Jβ)

    # generate all views
    ψall_l_views = map(x -> view(ψall_l, :, :, x), 1:arr_size)
    ψall_r_views = map(x -> view(ψall_r, :, :, x), 1:3)

    # left starter
    ψall_l_views[1] .= ψ0l
    if verbose >= 1
        println("$(typeof(ψw)), $(typeof(H)), $(typeof(ψ0l))")
    end
    mul!(ψw, H, ψ0l)
    ψall_l_views[2] .= ψw

    # right starter
    mul!(Jψ0r, Jα, ψ0r)

    reps = Integer(ceil(NC/arr_size - 1))
    println(typeof(H))
    println(typeof(Jα))
    println(typeof(ψall_r))
    println(typeof(ψall_l))


    for rep in 1:(reps+1)
        m1 = (rep - 1) * arr_size + 1
        m2 = min(rep * arr_size, NC)
        if verbose >= 1
            println("step $(rep)/$(reps+1): $(m1) to $(m2)")
        end
        rep_size = m2 - m1 + 1
        if mn_sym
            μ_rep_all = map(n -> view(μ, n, m1:m2), 1:m2)
        else
            μ_rep_all = map(n -> view(μ, n, m1:m2), 1:NC) # μ should be on host!!
        end
        # loop over l
        chebyshev_iter(H, ψall_l_views, rep_size)

        # loop over r
        n = 1
        ψall_r_views[n] .= Jψ0r
        mul!(JTnHJψr, Jβ,ψall_r_views[n])

        broadcast_dot_reduce_avg_2d_1d!(μ_rep_all[n], ψall_l_views, JTnHJψr, NR, rep_size; NC0=NC0(m1, n), NCstep=NCstep)
        ## TODO: IMPROVE THIS?

        # n = 2
        n = 2
        mul!(ψall_r_views[n], H, Jψ0r) # use initial values to calc Hψ0r
        mul!(JTnHJψr, Jβ, ψall_r_views[n])

        broadcast_dot_reduce_avg_2d_1d!(μ_rep_all[n], ψall_l_views, JTnHJψr, NR, rep_size; NC0=NC0(m1, n), NCstep=NCstep)

        if mn_sym
            n_enum = 3:m2
        else
            n_enum = 3:NC
        end
        if verbose >= 1
            n_enum = ProgressBar(n_enum)
        end
        for n in n_enum # TODO : save memory possible here. We do not need 3 vectors for psi 2
            chebyshev_iter_single(H,
                                  ψall_r_views[r_ipp(n)],
                                  ψall_r_views[r_ip(n)],
                                  ψall_r_views[r_i(n)])
            mul!(JTnHJψr, Jβ, ψall_r_views[r_i(n)])

            broadcast_dot_reduce_avg_2d_1d!(μ_rep_all[n], ψall_l_views, JTnHJψr, NR, rep_size; NC0=NC0(m1, n), NCstep=NCstep)
        end

        # wrap around to prepare for next
        chebyshev_iter_wrap(H, ψall_l_views, arr_size) #timed
    end

    if mn_sym
        # apply symmetry
        for m = 1:NC
            for n = (m + 1):NC
                μ[m, n] = real(μ[m, n])
                μ[n, m] = μ[m, n]
            end
        end
    end
    return nothing
end

#aliases
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
    psi_in = exp.(2im*pi*maybe_on_device_rand(dt_real, NH, NR))
    normalize_by_col(psi_in, NR)

    kpm_2d!(H, Jα, Jβ, NC, NR, NH, μ, psi_in; kwargs...)
    return nothing
end
### END OF ALIASES

function kpm_3d!(
                 H, Jα, Jβ, Jγ,
                 NC::Int64, NR::Int64, NH::Int64,
                 μ,
                 psi_in_l,
                 psi_in_r;
                 arr_size::Int64=3,
                 verbose=0,
                 # workspace kwargs
                 ψ0r = maybe_on_device_zeros(dt_cplx, NH, NR),
                 JTn1HJψr = maybe_on_device_zeros(dt_cplx, NH, NR),
                 ψall_r = maybe_on_device_zeros(dt_cplx, NH, NR, 3),
                 ψ0l = maybe_on_device_zeros(dt_cplx, NH, NR),
                 # workspace for sub problem (kpm_2d)
                 sub_ψ0r = maybe_on_device_zeros(dt_cplx, NH, NR),
                 sub_Jψ0r=maybe_on_device_zeros(dt_cplx, NH, NR),
                 sub_JTnHJψr=maybe_on_device_zeros(dt_cplx, NH, NR),
                 sub_ψall_r=maybe_on_device_zeros(dt_cplx, NH, NR, 3),
                 sub_ψ0l=maybe_on_device_zeros(dt_cplx, NH, NR),
                 sub_ψall_l=maybe_on_device_zeros(dt_cplx, NH, NR, arr_size),
                 sub_ψw=maybe_on_device_zeros(dt_cplx, NH, NR,)
                )
    println("Developing")

    # do not enforce normalization
    @assert (size(psi_in_r) == (NH, NR)) "`psi_in_r` has size $(size(psi_in_r)) but expecting $(NH), $(NR)"
    @assert (size(psi_in_l) == (NH, NR)) "`psi_in_l` has size $(size(psi_in_l)) but expecting $(NH), $(NR)"
    ψ0r .= maybe_to_device(psi_in_r)
    ψ0l .= maybe_to_device(psi_in_l)

    H = maybe_to_device(H)
    Jα = maybe_to_device(Jα)
    Jβ = maybe_to_device(Jβ)
    Jγ = maybe_to_device(Jγ)

    # generate all views
    ψall_r_views = map(x -> view(ψall_r, :, :, x), 1:3)
    μ_views = map(x -> view(μ, :, :, x), 1:NC) # Jα, n1, last index


    n1 = 1
    mul!(ψall_r_views[n1], Jα, ψ0r)
    kpm_2d!(
            H, Jβ, Jγ,
            NC, NR, NH,
            μ_views[n1],
            ψ0l, # psi_in_l
            ψall_r_views[r_i(n1)]; # psi_in_r
            arr_size=arr_size,
            ψ0r=sub_ψ0r, Jψ0r=sub_Jψ0r, JTnHJψr=sub_JTnHJψr,
            ψall_r=sub_ψall_r, ψ0l=sub_ψ0l, ψall_l=sub_ψall_l, ψw=sub_ψw
           )

    n1 = 2
    mul!(ψall_r_views[n1], H, ψall_r_views[r_ip(n1)])
    kpm_2d!(
            H, Jβ, Jγ,
            NC, NR, NH,
            μ_views[n1],
            ψ0l, # psi_in_l
            ψall_r_views[r_i(n1)]; # psi_in_r
            arr_size=arr_size,
            ψ0r=sub_ψ0r, Jψ0r=sub_Jψ0r, JTnHJψr=sub_JTnHJψr,
            ψall_r=sub_ψall_r, ψ0l=sub_ψ0l, ψall_l=sub_ψall_l, ψw=sub_ψw
           )

    for n1 in 3:NC
        if verbose >= 1
            println("n1=$(n1) out of $(NC)")
        end
        chebyshev_iter_single(H, ψall_r, r_ipp(n1), r_ip(n1), r_i(n1))
        kpm_2d!(
                H, Jβ, Jγ,
                NC, NR, NH,
                μ_views[n1],
                ψ0l, # psi_in_l
                ψall_r_views[r_i(n1)]; # psi_in_r
                arr_size=arr_size,
                ψ0r=sub_ψ0r, Jψ0r=sub_Jψ0r, JTnHJψr=sub_JTnHJψr,
                ψall_r=sub_ψall_r, ψ0l=sub_ψ0l, ψall_l=sub_ψall_l, ψw=sub_ψw
               )
    end

end


function kpm_3d(
                H, Jα, Jβ, Jγ,
                NC::Int64, NR::Int64, NH::Int64;
                arr_size::Int64=3,
                verbose=0,
                psi_in_l=nothing,
                psi_in_r=nothing,
                psi_in=nothing
               )
    μ = zeros(ComplexF64, NC, NC, NC)
    if !isnothing(psi_in)
        if (!isnothing(psi_in_l) || !isnothing(psi_in_r))
           @warn "`psi_in_l`, `psi_in_r` and `psi_in` are simoutaneously set. Taking `psi_in` and discarding the others"
       end
       psi_in_l = psi_in
       psi_in_r = psi_in
   elseif !isnothing(psi_in_l) || !isnothing(psi_in_r)
       if isnothing(psi_in_l) || isnothing(psi_in_r)
           @warn "only one of `psi_in_l` and `psi_in_r` is set. Setting them as the same."
           psi_in_l = something(psi_in_l, psi_in_r)
           psi_in_r = psi_in_l
       end
   else
       @info "Using random phase as random vector"
       psi_in_l = exp.(2pi * 1im * rand(NH, NR));
       KPM.normalize_by_col(psi_in_l, NR)
       psi_in_r = psi_in_l
   end

   kpm_3d!(H, Jα, Jβ, Jγ, NC, NR, NH, μ, psi_in_l, psi_in_r; arr_size=arr_size, verbose=verbose)
   return μ
end
