using KPM:hn, JacksonKernel, LorentzKernels, muND_apply_kernel_and_h, mu2D_apply_kernel_and_h

# test hn work as expected
@test hn(0) == 1
@test hn(1) == hn(2) == hn(3) == 2

NC=15
mu_1d = ones(NC)
mu_2d = ones(NC, NC)
mu_3d = ones(NC, NC, NC)

for kernel in [
               KPM.JacksonKernel,
               KPM.LorentzKernels(1.0),
               KPM.LorentzKernels(2.0),
               # add here when any new kernel is implemented
              ]

    mu_1d_tilde = KPM.muND_apply_kernel_and_h(mu_1d, NC, kernel; dims=[1])
    @test all(imag(mu_1d_tilde) .≈ 0) # stays real
    mu_1d_tilde = real(mu_1d_tilde)
    @test mu_1d_tilde[1] == 1 < mu_1d_tilde[2] # first term should be 1, and second term (after applying hn) should be larger than first 
    @test mu_1d_tilde[NC] < (2/NC) # all kernel should have very small last term, not larger than 1/NC *usually*
    @test all(mu_1d_tilde[2:end-1].>mu_1d_tilde[3:end]) # kernel should be damping by NC.


    mu_3d_tilde = KPM.muND_apply_kernel_and_h(mu_3d, NC, KPM.JacksonKernel; dims=[1,2,3])
    @test all(imag(mu_3d_tilde) .≈ 0) # stays real
    mu_3d_tilde = real(mu_3d_tilde)
    @test all(mu_3d_tilde[1,:,:] .== mu_3d_tilde[:,1,:] .== mu_3d_tilde[:,:,1]) # should be symmetric
    @test all(mu_3d_tilde[1,:,:]/mu_3d_tilde[1,1,1] .≈ mu_3d_tilde[2,:,:]/mu_3d_tilde[2,1,1]) 
end
