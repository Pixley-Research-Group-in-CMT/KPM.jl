module KPM

dt_real = Float32
dt_cplx = ComplexF32
dt_num = Union{Float32, ComplexF32}

include("utils/Utils.jl")
include("device.jl")


include("kernels/kernels.jl")

include("moment.jl")

include("applications/dos.jl")
include("applications/conductivity.jl")
include("applications/cpge.jl")

end # module
