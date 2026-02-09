using Test
using LinearAlgebra
using SparseArrays
using KPM

# Small correctness test for `chebyshev_iter`.
# Build a tiny Hermitian `H`, two starter vectors and run the
# recurrence both via the package and manually, then compare.

NH = 4
NR = 2
Niter = 6

Hrand = randn(ComplexF64, NH, NH) + 1im * randn(ComplexF64, NH, NH)
H = (Hrand + Hrand') / 2

ψ0 = randn(ComplexF64, NH, NR) + 1im * randn(ComplexF64, NH, NR)
ψ1 = randn(ComplexF64, NH, NR) + 1im * randn(ComplexF64, NH, NR)

ψall = zeros(ComplexF64, NH, NR, Niter)
ψall[:, :, 1] .= ψ0
ψall[:, :, 2] .= ψ1

# run package implementation (in-place) using array-of-views API
ψviews = map(i -> view(ψall, :, :, i), 1:Niter)
KPM.chebyshev_iter(H, ψviews, Niter)

# manual recurrence: T_n = 2 H T_{n-1} - T_{n-2}
expected = Vector{Array{ComplexF64,2}}(undef, Niter)
expected[1] = copy(ψ0)
expected[2] = copy(ψ1)
for i in 3:Niter
	expected[i] = 2 * H * expected[i-1] - expected[i-2]
end

@testset "chebyshev_iter correctness" begin
	for i in 1:Niter
		@test norm(ψall[:, :, i] .- expected[i]) < 1e-10
	end
end
