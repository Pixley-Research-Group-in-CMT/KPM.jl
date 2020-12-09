module KPM

dt_real = Float64
dt_cplx = ComplexF64
dt_num = Union{Float64, ComplexF64}

include("device.jl")

include("utils/physics.jl")
include("utils/external.jl")
include("utils/vectors.jl")
include("utils/chebyshev.jl")

include("kernels/kernels.jl")

include("moment.jl")

include("applications/dos.jl")
include("applications/conductivity.jl")

end # module
