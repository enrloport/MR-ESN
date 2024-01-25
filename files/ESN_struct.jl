Base.@kwdef mutable struct ESN
    R         ::Mtx     = zeros(1,1)
    R_in      ::Mtx     = zeros(1,1)
    R_fdb     ::Mtx     = zeros(1,1)
    R_out     ::Mtx     = zeros(1,1)
    Y         ::Mtx     = zeros(1,1)
    X         ::Mtx     = zeros(1,1)
    x         ::Mtx     = zeros(1,1)
    R_size    ::Int16   = size(R,1)
    R_scaling ::Float64 = 1.0
    alpha     ::Float64 = 0.5
    beta      ::Float64 = 1.0e-8
    rho       ::Float64 = 1.0
    sigma     ::Float64 = 1.0
    hadamard  ::Bool    = false
    sgmd      ::Function= tanh
    F_in      ::Function= (f,u) -> hadamard ? R_in .* f(u) : R_in * f(u)
end

Base.@kwdef mutable struct MrESN
    esns            ::Array{Any}
    train_function  ::Function      = __do_train_MrESN!
    test_function   ::Function      = __do_test_MrESN!
    X               ::Mtx           = zeros(1,1)
    R_out           ::Mtx           = zeros(1,1)
    beta            ::Float64       = 1.0e-8
    error           ::Float64       = 1.0
    wrong_class     ::Array{Any}    = []
    classes_Y       ::Array{Any}    = []
    Y_target        ::Array{Any}    = []
    Y               ::Array{Any}    = []
    classes_Routs   ::Dict{Int16,Union{Array{Float64},CuArray}} = Dict()
end



Base.@kwdef mutable struct deepESN
    esns            ::Array{Any}
    train_function  ::Function      = __do_train_deepESN!
    test_function   ::Function      = __do_test_deepESN!
    X               ::Mtx           = zeros(1,1)
    R_out           ::Mtx           = zeros(1,1)
    beta            ::Float64       = 1.0e-8
    error           ::Float64       = 1.0
    wrong_class     ::Array{Any}    = []
    classes_Y       ::Array{Any}    = []
    Y_target        ::Array{Any}    = []
    Y               ::Array{Any}    = []
    classes_Routs   ::Dict{Int16,Union{Array{Float64},CuArray}} = Dict()
end