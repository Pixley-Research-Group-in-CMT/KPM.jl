module KPM

dt_real = Float32
dt_cplx = Complex{Float32}
dt_num = Union{Float32, Float32}

include("device.jl")

include("utils/Utils.jl")

include("kernels/kernels.jl")

include("moment.jl")

include("applications/dos.jl")
include("applications/conductivity.jl")
include("applications/cpge.jl")

end # module
