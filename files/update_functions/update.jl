
function update(esn,u, f)
    # println("R_in", typeof(esn.R_in) , " size: ", size(esn.R_in))
    # println("u", typeof(u) , " size: ", size(u))
    # println("R", typeof(esn.R) , " size: ", size(esn.R_in))
    # println("x", typeof(esn.x) , " size: ", size(esn.x))
    esn.x[:] = (1-esn.alpha).*esn.x .+ esn.alpha.*esn.sgmd.( esn.F_in(f,u) .+ esn.R*esn.x)
end