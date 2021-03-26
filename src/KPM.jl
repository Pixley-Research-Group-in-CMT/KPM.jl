module KPM

dt_real = Float16
dt_cplx = ComplexF16
dt_num = Union{Float16, ComplexF16}

include("device.jl")

include("utils/Utils.jl")

include("kernels/kernels.jl")

include("moment.jl")

include("applications/dos.jl")
include("applications/conductivity.jl")
include("applications/cpge.jl")

end # module
