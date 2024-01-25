include("../../ESN.jl")
using MLDatasets
using CUDA
CUDA.allowscalar(false)
using Wandb


# Random.seed!(42)

# FashionMNIST dataset
train_x, train_y = FashionMNIST(split=:train)[:]
test_x, test_y = FashionMNIST(split=:test)[:]

repit =1 #10
_params = Dict{Symbol,Any}(
     :gpu               => true
    ,:wb                => false
    ,:wb_logger_name    => "MRESN_tanh_fashion_GPU"
    ,:classes           => [0,1,2,3,4,5,6,7,8,9]
    ,:beta              => 1.0e-10
    ,:train_length      => size(train_y)[1]
    ,:test_length       => size(test_y)[1]
    ,:initial_transient => 2
    ,:train_f           => __do_train_MrESN_mnist!
    ,:test_f            => __do_test_MrESN_mnist!
)


for _ in 1:repit
    sd = rand(1:10000)
    Random.seed!(sd)
    _params[:num_esns] = 4
    min_d, max_d = 0.1, 0.25
    _params_esn = Dict{Symbol,Any}(
        :R_scaling => rand(Uniform(0.5,1.5),_params[:num_esns])
        ,:alpha    => [0.7 for _ in 1:_params[:num_esns]] #rand(Uniform(0.5,1.0),_params[:num_esns])
        ,:density  => rand(Uniform(min_d, max_d),_params[:num_esns])
        ,:rho      => rand(Uniform(0.5,1.5),_params[:num_esns])
        ,:sigma    => rand(Uniform(0.5,1.5),_params[:num_esns])
        ,:nodes    => [ 500 for _ in 1:_params[:num_esns] ] # rand([500, px*px ,1000],_params[:num_esns])
        ,:sgmds    => [ tanh for _ in 1:_params[:num_esns] ]
    )
    _params[:image_size]   = (28,28)
    _params[:train_data]   = train_x
    _params[:test_data]    = test_x
    _params[:train_labels] = train_y
    _params[:test_labels]  = test_y
    par = Dict(
        "Reservoirs" => _params[:num_esns]
        , "Total nodes"        => sum(_params_esn[:nodes])
        , "Train length"       => _params[:train_length]
        , "Test length"        => _params[:test_length]
        , "Resized"            => _params[:image_size][1]
        , "Nodes per reservoir"=> _params_esn[:nodes]
        , "Initial transient"  => _params[:initial_transient]
        , "seed"               => sd
        , "sgmds"              => _params_esn[:sgmds]
        , "alphas"             => _params_esn[:alpha]
        , "beta"               => _params[:beta]
	    , "densities"          => _params_esn[:density]
        , "max_density"        => max_d
	    , "min_density"        => min_d
	    , "rhos"               => _params_esn[:rho]
        , "sigmas"             => _params_esn[:sigma]
        , "R_scalings"         => _params_esn[:R_scaling]
        , "Constant term"      => 1 # _params[:num_esns]
	    , "preprocess"         => "yes"
    )
    if _params[:wb]
        using Logging
        using Wandb
        _params[:lg] = wandb_logger(_params[:wb_logger_name])
        Wandb.log(_params[:lg], par )
    else
        display(par)
    end
    par = Dict(""=>0)
    GC.gc()

    tm = @elapsed begin
        r1 = do_batch_mresn(_params_esn,_params, sd)
    end
    if _params[:wb]
        close(_params[:lg])
    end
    println("Error: ", r1.error )
    if _params[:gpu]
        println("Time GPU: ", tm )
    else
        println("Time CPU: ", tm )
    end
end


# EOF
