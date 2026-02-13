module KPM

dt_real = Float64
dt_cplx = ComplexF64
dt_num = Union{Float64, ComplexF64}

try
	import CUDA
        include(joinpath(@__DIR__, "..", "ext", "KPMCUDAExt.jl"))
catch
	# CUDA not available — leave CUDA extension unloaded
end

include("device.jl")

include("utils/Utils.jl")

include("kernels/kernels.jl")

include("moment.jl")

include("applications/dos.jl")
include("applications/ldos_mu.jl")
include("applications/conductivity.jl")
include("applications/cpge.jl")
include("applications/optical_cond.jl")

end # module
