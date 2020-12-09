var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = KPM","category":"page"},{"location":"#KPM","page":"Home","title":"KPM","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [KPM]","category":"page"},{"location":"#KPM.JacksonKernel-Tuple{Integer,Integer}","page":"Home","title":"KPM.JacksonKernel","text":"JacksonKernel(n::Integer, N::Integer)\n\nJacksonkernel evaluated at n-th expansion coefficient with N in total (NC)\n\n\n\n\n\n","category":"method"},{"location":"#KPM.LorentzKernels-Tuple{Float64}","page":"Home","title":"KPM.LorentzKernels","text":"LorentzKernels(λ::Float64)\n\nReturns function LorentzKernel(n, N) that evaluates Lorentz kernel with parameter λ, at n-th expansion coefficient with N in total (NC)\n\n\n\n\n\n","category":"method"},{"location":"#KPM.broadcast_dot_1d_1d!","page":"Home","title":"KPM.broadcast_dot_1d_1d!","text":"Vl and Vr are both [NH, NR] sized array. Each corresponding [:, NR] slice pair is dotted, saving into the target of [NR], multiplying by alpha and plus beta. Beta is either a number or vector of [NR]\n\n\n\n\n\n","category":"function"},{"location":"#KPM.broadcast_dot_2d_1d!-Tuple{Array{T,2} where T,Array{T,3} where T,Array{T,2} where T,Int64,Int64}","page":"Home","title":"KPM.broadcast_dot_2d_1d!","text":"Dot product each column of Vls with vector Vr, save in target.\n\ntarget: 2D Array (n, NR), n >= ncols. Vls: 3D Array, shape (NH, NR, n), where n >= ncols. Vr: 2D Array, shape NH, NR ncols: Integer, number of columns. \n\n\n\n\n\n","category":"method"},{"location":"#KPM.broadcast_dot_reduce_avg_2d_1d!-Tuple{Array{T,1} where T,Array{T,1} where T<:(SubArray{Ts,2,P,I,L} where L where I where P where Ts),Array{T,2} where T,Int64,Int64}","page":"Home","title":"KPM.broadcast_dot_reduce_avg_2d_1d!","text":"Dot product each column of Vls with vector Vr, save in target. Each view has NR replica of NH. This function take the average.\n\ntarget: 1D Array (n), n >= ncols. Vls: 1D Array of 2D views, shape (n), each view (NH, NR), where n >= ncols. Vr: 2D Array, shape NH, NR ncols: Integer, number of columns. \n\n\n\n\n\n","category":"method"},{"location":"#KPM.chebyshev_iter-Tuple{Any,Union{CUDA.CuArray{T,2}, Array{T,2}} where T,Int64}","page":"Home","title":"KPM.chebyshev_iter","text":"evaluate from 3 to n ψall[:,3] comes from ψall[:,2] and ψall[:,1] separating into two function might improve performance (or not???)\n\n\n\n\n\n","category":"method"},{"location":"#KPM.dos","page":"Home","title":"KPM.dos","text":"dos(μ, a, Ntilde; Erange, T, T, NC, kernel)\n\ndefined at /home/runner/work/KPM.jl/KPM.jl/src/applications/dos.jl:17.\n\ndos(μ, a::Float64, Ntilde::Integer; Erange::Array{Float64,1}=[-1.0,1.0], NC::Integer=0, kernel=JacksonKernel)\n\nCalculate DOS for a fermi energy grid spanning Erange with Ntilde total points. If Erange is (0,0), automatically set it to be sightly smaller than full size.\n\n\n\n\n\n","category":"function"},{"location":"#KPM.fermiFunction-Tuple{Float64,Float64,Float64}","page":"Home","title":"KPM.fermiFunction","text":"fermiFunction(E, E_f, beta)\n\ncalculate Fermi-Dirac function at energy E, Fermi energy μ and temperature β =1/T. Input and output all Float64. Infinite β only allowed when accessing fermi energy through fermiFunctions(). [For performance reason for now. TODO: allow β=Inf here withouth perf. reduction. ]\n\nAllow sloppy use of type as long as convertion is available, if using keyword arguments. \n\n\n\n\n\n","category":"method"},{"location":"#KPM.fermiFunctions-Tuple{Float64,Float64}","page":"Home","title":"KPM.fermiFunctions","text":"fermiFunctions(E_f::Float64, beta::Float64)\n\nreturns a fermi function with given E_f and beta. \n\nAllow sloppy use of type as long as convertion is available, if using keyword arguments. \n\n\n\n\n\n","category":"method"},{"location":"#KPM.kpm_1d","page":"Home","title":"KPM.kpm_1d","text":"kpm_1d(H, NC, NR, NH; psi_in, psi_in_l, psi_in_r, force_norm, verbose, avg_output)\n\ndefined at /home/runner/work/KPM.jl/KPM.jl/src/moment.jl:179.\n\nThe simple version of 1D KPM that returns the moment.\n\nH           – Hamiltonian. A matrix or sparse matrix\nNC          – Integer. the cut off dimension\nNR          – Integer. number of random vectors used for KPM evaluation\nNH          – Integer. the size of hamiltonian\npsi_in      – Optional. Allow setting random vector manually.\nforce_norm  – Boolean, Optional. Apply normalization.\nverbose     – Integer. Default is 0. Enables progress bar if set verbose=1.\navg_output  – Boolean. Default is true. Whether to output averaged μ (hence size NC) or separate μs (size NR x NC).\n\n\n\n\n\n","category":"function"},{"location":"#KPM.kpm_1d!","page":"Home","title":"KPM.kpm_1d!","text":"kpm_1d!(H, NC, NR, NH, mu_all, psi_in; verbose, α_all)\n\ndefined at /home/runner/work/KPM.jl/KPM.jl/src/moment.jl:219.\n\nkpm_1d!(H, NC, NR, NH, mu; kwargs...)\n\ndefined at /home/runner/work/KPM.jl/KPM.jl/src/moment.jl:278.\n\nkpm_1d!(H, NC, NR, NH, mu, psi_in_l, psi_in_r; kwargs...)\n\ndefined at /home/runner/work/KPM.jl/KPM.jl/src/moment.jl:287.\n\nThe in-place version of 1D KPM.  Calculate the moments μ defined in KPM. Output is saved in mu.\n\nH           – Hamiltonian. A matrix or sparse matrix.\nNC          – Integer. the cut off dimension.\nNR          – Integer. number of random vectors used for KPM evaluation.\nNH          – Integer. the size of hamiltonian.\nmu_all          – Array. Output for each random vector. Size (NR, NC). \npsi_in      – Array (optional). Input array on the right side. A ket.\n\n\n\n\n\n","category":"function"},{"location":"#KPM.kpm_2d","page":"Home","title":"KPM.kpm_2d","text":"kpm_2d(H, Ja, Jb, NC, NR, NH; psi_in, psi_in_l, psi_in_r, kwargs...)\n\ndefined at /home/runner/work/KPM.jl/KPM.jl/src/moment.jl:299.\n\nThe simple version of 2D KPM that returns the moment. Calculate moments for 2D KPM. \n\nCalculates ψ0l * Tm(H) * Jβ * Tn(H) * Jα * ψ0r. When ψ0r and ψ0l are chosen to be random and identical, the output approximates tr(Tm(H) Jβ Tn(H) Jα). The accuracy is ~ O(1/sqrt(NR * NH)). NC controls the energy resolution of the result.\n\nOutput: μ, a 2D array in ComplexF64. μ[n, m] is the momentum for 2D KPM.\n\nARGS\n\nH\n\nHamiltonian. A sparse 2D array.\n\nJα\n\nCurrent operator. A sparse 2D array.\n\nJβ\n\nCurrent operator. A sparse 2D array.\n\nNC\n\nInteger. KPM cutoff order.\n\nNR\n\nInteger. Number of random vectors to choose from. When skipped, understood as NR=1.\n\nNH\n\nInteger. Dimension of H, Jα and Jβ\n\nKWARGS\n\npsi_in_l\n\nPasses value to ψ0l. The array is not updated. Size should be (NH, NR) (preferred) or (NR, NH) if set.\n\npsi_in_r\n\nPasses value to ψ0r. The array is not updated. Size should be  (NH, NR) (preferred) or (NR, NH) if set.\n\npsi_in\n\nCannot be used together with psiinl and psiinr. Sets psiinl=psiinr=psi_in if set.\n\nkwargs\n\nother kwargs in KPM_2D!\n\n\n\n\n\n","category":"function"},{"location":"#KPM.kpm_2d!","page":"Home","title":"KPM.kpm_2d!","text":"kpm_2d!(H, Jα, Jβ, NC, NR, NH, μ, psi_in_l, psi_in_r; arr_size, verbose, ψ0r, Jψ0r, JTnHJψr, ψall_r, ψ0l, ψall_l, ψw)\n\ndefined at /home/runner/work/KPM.jl/KPM.jl/src/moment.jl:310.\n\nkpm_2d!(H, Jα, Jβ, NC, NR, NH, μ, psi_in; kwargs...)\n\ndefined at /home/runner/work/KPM.jl/KPM.jl/src/moment.jl:402.\n\nkpm_2d!(H, Jα, Jβ, NC, NR, NH, μ; kwargs...)\n\ndefined at /home/runner/work/KPM.jl/KPM.jl/src/moment.jl:412.\n\nIn place KPM2D. This is also the main building block for KPM_2D. This method only provide NR=1.\n\nCalculates ψ0l * Tm(H) * Jβ * Tn(H) * Jα * ψ0r.  When ψ0r and ψ0l are chosen to be random and identical, the output approximates tr(Tm(H) Jβ Tn(H) Jα). The accuracy is sim O(1sqrt(NR * NH)) with NR repetitions. NC controls the energy resolution of the result.\n\nOutput: nothing. Result is saved on μ.\n\nARGS\n\nH : Hamiltonian. A sparse 2D array.\nJα : Current operator. A sparse 2D array.\nJβ : Current operator. A sparse 2D array.\nNC : Integer. KPM cutoff order.\nNR : Integer. Number of random vectors.\nNH : Integer. Dimension of H, Jα and Jβ\nμ : 2D Array of dimension (NC, NC). Results will be updated here. Any data\n\nwill be overwritten.\n\npsi_in : Set psi_in_l and psi_in_r. Size is (NH, NR). The array is not updated.\n\nWhether the input is normalized or not, it is assumed to be intended. Usually psi_in should be normalized.\n\npsi_in_l : Passes value to ψ0l. Size is (NH, NR). The array is not updated.\n\nWhether the input is normalized or not, it is assumed to be intended. Usually psi_in_l should be normalized. psi_in_l is given as column vector of ket ψl = ψl^dagger\n\npsi_in_r : Passes value to ψ0r. Size is (NH, NR). The array is not updated.\n\nWhether the input is normalized or not, it is assumed to be intended. Usually psi_in_r should be normalized. psi_in_r is given as column vector of ket ψr. \n\nKWARGS\n\narr_size : The buffer array size. Minimum is 3. Determines the number of\n\nleft states to be kept in memory for each loop of right states. The time complexity is reduced from O(Ntimes NC^2) to O(Ntimes NCtimes arr_size) while space complexity is increased from O(Ntimes NC) to O(Ntimes NCtimes arr_size).\n\nworking spaces KWARGS: The following keyword args are simply providing working place arrays to avoid repetitive allocation and GC. They are automatically created if not set. However, when using KPM_2D! for many times, it is beneficial to reuse those arrays.  CONVENTION: args with ψ are all working space arr.\n\nψ0r=maybe_on_device_zeros(NH, NR)\nJψ0r=maybe_on_device_zeros(NH, NR)\nJTnHJψr=maybe_on_device_zeros(NH, NR)\nψall_r=maybe_on_device_zeros(3, NH, NR)\nψ0l=maybe_on_device_zeros(NH, NR)\nψall_l=maybe_on_device_zeros(arr_size, NH, NR)\nψw=maybe_on_device_zeros(NH, NR)\n\n\n\n\n\n","category":"function"},{"location":"#KPM.normalize_by_col-Tuple{Any,Any}","page":"Home","title":"KPM.normalize_by_col","text":"Normalize a collection of vectors in an (NH, NR) array psi_in, where each column (that is psi_in[:, NRi]) represent a separate vector.\n\n\n\n\n\n","category":"method"}]
}
