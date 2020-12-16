"""
Utility for processing test data.
Should prepare a "ops_small.jld2"  file as data at first, then use
this utility to create "ops_small_ED.jld2" that includes cumulated
DOS calculated out of exact diagonalization.
Used for comparison with KPM dos under integration test
"""

using JLD2
using LinearAlgebra
@load "ops_small.jld2"

H = Matrix(H_norm * a)

ev, es = eigen(H)
println(size(ev))
NH = size(H)[1]

E_grid = collect(-2:0.01:2)
c_dos_ED = 0 * E_grid

for (i, E) in enumerate(E_grid)
    c_dos[i] = sum(ev <= E) / NH
end

@save "ops_small_ED.jld2", E_grid, c_dos_ED
