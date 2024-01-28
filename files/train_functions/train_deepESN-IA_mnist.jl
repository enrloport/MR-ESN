function __fill_X_deepESNIA_mnist!(deepE, args::Dict )

    f = args[:gpu] ? (u) -> CuArray(reshape(u, :, 1)) : (u) -> reshape(u, :, 1)

    function __step(t,fu,f)
        update(deepE.esns[1], args[:train_data][:,:,t], f )

        for i in 2:length(deepE.esns)
            update(deepE.esns[i], vcat(deepE.esns[i-1].x, fu ), f )
        end
    end

    for t in 1:args[:train_length]
        fu = f(args[:train_data][:,:,t])
        for _ in 1:args[:initial_transient]
            __step(t,fu,f)
        end
        __step(t,fu,f)
        deepE.X[:,t] = vcat(fu, [_e.x for _e in deepE.esns]... , f([1]) )
        # deepE.X[:,t] = vcat(fu, deepE.esns[end].x, f([1]) )
        for es in deepE.esns
            es.x[:] = es.x .* 0
        end
    end
end


function __make_Rout_deepESNIA_mnist!(deepE,args)
    X             = deepE.X
    classes       = args[:classes]
    classes_Yt    = Dict( c => zeros(args[:train_length]) for c in classes )  # New dataset for each class

    for t in 1:args[:train_length]
        lt = args[:train_labels][t]
        for c in classes
            y = lt == c ? 1.0 : 0.0
            classes_Yt[c][t] = y
        end
    end
    if args[:gpu]
        classes_Yt = Dict( k => CuArray(classes_Yt[k]) for k in keys(classes_Yt) )
    end

    cudamatrix = args[:gpu] ? CuArray : Matrix
    deepE.classes_Routs = Dict( c => cudamatrix(transpose((X*transpose(X) + deepE.beta*I) \ (X*classes_Yt[c]))) for c in classes )
end


function __do_train_deepESNIA_mnist!(deepE, args)
    num   = args[:train_length]
    # deepE.X = zeros( deepE.esns[end].R_size + args[:image_size][1]*args[:image_size][2] + 1, num )
    deepE.X = zeros( sum([esn.R_size for esn in deepE.esns]) + args[:image_size][1]*args[:image_size][2] + 1, num)
    #deepE.X = zeros( sum([esn.R_size for esn in deepE.esns]) + args[:image_size][1]*args[:image_size][2] + length(deepE.esns), num)
    for i in 1:length(deepE.esns)
        deepE.esns[i].x = zeros( deepE.esns[i].R_size, 1)
    end

    if args[:gpu]
        deepE.X = CuArray(deepE.X)
        for i in 1:length(deepE.esns)
            deepE.esns[i].x = CuArray(deepE.esns[i].x)
        end
    end

    __fill_X_deepESNIA_mnist!(deepE,args)
    __make_Rout_deepESNIA_mnist!(deepE,args)
end
