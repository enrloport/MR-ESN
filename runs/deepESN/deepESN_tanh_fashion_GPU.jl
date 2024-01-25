include("../../ESN.jl")
using MLDatasets
using CUDA
CUDA.allowscalar(false)
using Wandb


# Random.seed!(42)

# FashionMNIST dataset
train_x, train_y = FashionMNIST(split=:train)[:]
test_x, test_y = FashionMNIST(split=:test)[:]

repit = 1 #100
_params = Dict{Symbol,Any}(
     :gpu           => true
    ,:wb            => false
    ,:wb_logger_name=> "deepESN_tanh_fashion_GPU"
    ,:classes       => [0,1,2,3,4,5,6,7,8,9]
    ,:beta          => 1.0e-8
    ,:initial_transient=>2
    ,:train_length  => size(train_y)[1] -55000
    ,:test_length   => size(test_y)[1] -9000
    ,:train_f       => __do_train_deepESN_mnist!
    ,:test_f        => __do_test_deepESN_mnist!
)


px      = 28 # rand([14,20,25,28])
sz      = (px,px)



function do_batch_deepESN(_params_esn, _params,sd)
    sz       = _params[:image_size]
    im_sz    = sz[1]*sz[2] 
    nodes    = _params_esn[:nodes] #im_sz
    rhos     = _params_esn[:rho]
    sigmas   = _params_esn[:sigma]
    sgmds    = _params_esn[:sgmds]
    densities= _params_esn[:density]
    alphas   = _params_esn[:alpha]
    r_scales = _params_esn[:R_scaling]

    esn1 = ESN( 
    #     R      = _params[:gpu] ? CuArray(new_R(nodes[1], density=densities[1], rho=rhos[1])) : new_R(nodes[1], density=densities[1], rho=rhos[1])
    #    ,R_in   = _params[:gpu] ? CuArray(rand(Uniform(-sigmas[1],sigmas[1]), nodes[1], im_sz )) : rand(Uniform(-sigmas[1],sigmas[1]), nodes[1], im_sz )
        R =  new_R(nodes[1], density=densities[1], rho=rhos[1], gpu = _params[:gpu])
        ,R_in = new_R_in(nodes[1], im_sz, sigma=sigmas[1], gpu=_params[:gpu])
        ,R_scaling = r_scales[1], alpha = alphas[1], rho = rhos[1], sigma = sigmas[1], sgmd = sgmds[1]
    )
    esn2 = [
        ESN( 
            #  R      = _params[:gpu] ? CuArray(new_R(nodes[i], density=densities[i], rho=rhos[i])) : new_R(nodes[i], density=densities[i], rho=rhos[i])
            # ,R_in   = _params[:gpu] ? CuArray(rand(Uniform(-sigmas[i],sigmas[i]), nodes[i], nodes[i-1] )) : rand(Uniform(-sigmas[i],sigmas[i]), nodes[i], nodes[i-1] )
            R =  new_R(nodes[i], density=densities[i], rho=rhos[i], gpu = _params[:gpu])
            ,R_in = new_R_in(nodes[i], im_sz, sigma=sigmas[i], gpu=_params[:gpu])
            ,R_scaling = r_scales[i]
            ,alpha  = alphas[i]
            ,rho    = rhos[i]
            ,sigma  = sigmas[i]
            ,sgmd   = sgmds[i]
        ) for i in 2:_params[:num_esns]
    ]

    tms = @elapsed begin
        deepE = deepESN(
            esns=[esn1, esn2...]
            ,beta=_params[:beta] 
            ,train_function = _params[:train_f]
            ,test_function = _params[:test_f]
            )
        tm_train = @elapsed begin
            deepE.train_function(deepE,_params)
        end
        tm_test = @elapsed begin
            deepE.test_function(deepE,_params)
        end
    end
 
    to_log = Dict(
        "Total time"  => tms
        ,"Train time"=> tm_train
        ,"Test time"=> tm_test
       , "Error"    => deepE.error
    )
    if _params[:wb] 
        Wandb.log(_params[:lg], to_log )
    else
        display(to_log)
    end
    return deepE
end


for _ in 1:repit
    for num_nodes in [500]
        for num_esns in [2]
            _params[:num_R_nodes] = num_nodes
            _params[:num_esns] = num_esns
            sd = rand(1:10000)
            Random.seed!(sd)
            _params_esn = Dict{Symbol,Any}(
                :R_scaling => rand(Uniform(0.5,1.5),_params[:num_esns])
                ,:alpha    => [0.7 for _ in 1:_params[:num_esns]]
                ,:density  => rand(Uniform(0.01,0.2),_params[:num_esns])
                ,:rho      => rand(Uniform(0.5,1.5),_params[:num_esns])
                ,:sigma    => rand(Uniform(0.5,1.5),_params[:num_esns])
                ,:nodes    => [_params[:num_R_nodes] for _ in 1:_params[:num_esns] ] # rand([500, px*px ,1000],_params[:num_esns])
                ,:sgmds    => [tanh for _ in 1:_params[:num_esns] ]
            )
            _params[:image_size]   = sz
            _params[:train_data]   = train_x
            _params[:test_data]    = test_x
            _params[:train_labels] = train_y
            _params[:test_labels]  = test_y
            par = Dict(
                "Layers" => _params[:num_esns]
                , "Total nodes"  => sum(_params_esn[:nodes])
                , "Train length" => _params[:train_length]
                , "Test length"  => _params[:test_length]
                , "Resized"      => _params[:image_size][1]
                , "Nodes per layer"=> _params_esn[:nodes]
                , "Initial transient"=> _params[:initial_transient]
                , "seed"         => sd
                , "sgmds"        => _params_esn[:sgmds]
                , "alphas" => _params_esn[:alpha]
                , "densities" => _params_esn[:density]
                , "rhos" => _params_esn[:rho]
                , "sigmas" => _params_esn[:sigma]
                , "R_scalings" => _params_esn[:R_scaling]
                )
            if _params[:wb]
                using Logging
                using Wandb
                _params[:lg] = wandb_logger(_params[:wb_logger_name])
                Wandb.log(_params[:lg], par )
            end
            display(par)
        
            r1=[]
            tm = @elapsed begin
                r1 = do_batch_deepESN(_params_esn,_params, sd)
            end
            if _params[:wb]
                close(_params[:lg])
            end
        
            printime = _params[:gpu] ? "Time GPU: " * string(tm) :  "Time CPU: " * string(tm) 
            println("Error: ", r1.error, "\n", printime  )
        
        end
    end 
end




