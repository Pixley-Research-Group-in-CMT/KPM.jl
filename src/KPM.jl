module KPM

dt_real = Float64
dt_cplx = Complex{Float64}
dt_num = Union{Float64, Float64}

include("device.jl")

include("utils/Utils.jl")

include("kernels/kernels.jl")

include("moment.jl")

include("applications/dos.jl")
include("applications/conductivity.jl")
include("applications/cpge.jl")

end # module
