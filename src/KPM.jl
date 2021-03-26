module KPM
using BFloat16s

dt_real = BFloat16
dt_cplx = Complex{BFloat16}
dt_num = Union{BFloat16, BFloat16}

include("device.jl")

include("utils/Utils.jl")

include("kernels/kernels.jl")

include("moment.jl")

include("applications/dos.jl")
include("applications/conductivity.jl")
include("applications/cpge.jl")

end # module
