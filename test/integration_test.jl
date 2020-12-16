using JLD2
using Test
using KPM

@load "test_data/ops_small.jld2"
@load "test_data/exact_dos.jld2"
H_orig = H_norm * a
E, dos = KPM.dos(H_orig)


