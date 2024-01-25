
function update(esn,u, f)
    # println(typeof(esn.R_in))
    # println(typeof(u))
    # println(typeof(esn.R))
    # println(typeof(esn.x))
    esn.x[:] = (1-esn.alpha).*esn.x .+ esn.alpha.*esn.sgmd.( esn.F_in(f,u) .+ esn.R*esn.x)
end