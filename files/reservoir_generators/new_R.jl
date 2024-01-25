

function new_R( R_size::Int=50; R_scaling::Float64=1.0, rho::Float64=1.0, density=1.0, distribution=Uniform, gpu=false, bounds=(0,0))
    up,low = -R_scaling, R_scaling
    if bounds[1] + bounds[2] != 0
        up,low = bounds[1], bounds[2]
    end
    if density != 1.0
        W = sprand(R_size, R_size, density, x-> rand(distribution(up, low), x) )
        W = Array(W)
    else
        W = rand( distribution( up, low ) , R_size, R_size )
    end
    set_spectral_radius!( W , rho)

    if gpu W = CuArray(W) end
    return W
end