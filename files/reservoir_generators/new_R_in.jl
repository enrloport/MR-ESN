

function new_R_in(m, n; sigma=1.0, density=1.0, gpu=false, distribution=Uniform, channels=1)
    R_in = sprand(m, n, density, x-> rand(distribution(-sigma, sigma), x) )
    R_in = hcat([R_in for _ in 1:channels]...)
    if gpu R_in = CuArray(R_in) end
    return R_in
end
