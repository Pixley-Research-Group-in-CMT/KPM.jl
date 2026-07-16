module KPM

const dt_real = Float64
const dt_cplx = ComplexF64
const dt_num = Union{Float64, ComplexF64}

include("device.jl")

include("utils/Utils.jl")

include("kernels/kernels.jl")

include("moment.jl")

include("applications/dos.jl")
include("applications/ldos_mu.jl")
include("applications/conductivity.jl")
include("applications/cpge.jl")
include("applications/optical_cond.jl")
include("frontend.jl")

end # module
