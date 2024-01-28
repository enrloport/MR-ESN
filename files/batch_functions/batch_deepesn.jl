
function do_batch_deepESN(_params_esn, _params,sd)
    sz       = _params[:image_size]
    im_sz    = sz[1]*sz[2] 
    nodes    = _params_esn[:nodes]
    rhos     = _params_esn[:rho]
    sigmas   = _params_esn[:sigma]
    sgmds    = _params_esn[:sgmds]
    densities= _params_esn[:density]
    alphas   = _params_esn[:alpha]
    r_scales = _params_esn[:R_scaling]

    esn1 = ESN( 
        R =  new_R(nodes[1], density=densities[1], rho=rhos[1], gpu = _params[:gpu])
        ,R_in = new_R_in(nodes[1], im_sz, sigma=sigmas[1], gpu=_params[:gpu])
        ,R_scaling = r_scales[1], alpha = alphas[1], rho = rhos[1], sigma = sigmas[1], sgmd = sgmds[1]
    )
    esn2 = [
        ESN( 
            R =  new_R(nodes[i], density=densities[i], rho=rhos[i], gpu = _params[:gpu])
            ,R_in = new_R_in(nodes[i], nodes[i-1], sigma=sigmas[i], gpu=_params[:gpu])
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

