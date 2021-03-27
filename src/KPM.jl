module KPM

dt_real = Float16
dt_cplx = Complex{Float16}
dt_num = Union{Float16, Float16}

include("device.jl")

include("utils/Utils.jl")

include("kernels/kernels.jl")

include("moment.jl")

include("applications/dos.jl")
include("applications/conductivity.jl")
include("applications/cpge.jl")

end # module
