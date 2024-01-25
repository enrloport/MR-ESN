function __fill_X_MrESN_mnist!(mrE, args::Dict )

    f = args[:gpu] ? (u) -> CuArray(reshape(u, :, 1)) : (u) -> reshape(u, :, 1)

    for t in 1:args[:train_length]
        for i in 1:length(mrE.esns)
            for _ in 1:args[:initial_transient]
                update(mrE.esns[i], args[:train_data][:,:,t], f )
            end
            update(mrE.esns[i], args[:train_data][:,:,t], f )
        end
        mrE.X[:,t] = vcat(f(args[:train_data][:,:,t]),[es.x for es in mrE.esns]...,f([1 for _ in 1:1 ]))
        #mrE.X[:,t] = vcat(f(args[:train_data][:,:,t]),[es.x for es in mrE.esns]...,f([1 for _ in 1:length(mrE.esns) ]))
    end
end


function __make_Rout_MrESN_mnist!(mrE,args)
    X             = mrE.X
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
    mrE.classes_Routs = Dict( c => cudamatrix(transpose((X*transpose(X) + mrE.beta*I) \ (X*classes_Yt[c]))) for c in classes )
end


function __do_train_MrESN_mnist!(mrE, args)
    num   = args[:train_length]
    mrE.X = zeros( sum([esn.R_size for esn in mrE.esns]) + args[:image_size][1]*args[:image_size][2] + 1, num)
    #mrE.X = zeros( sum([esn.R_size for esn in mrE.esns]) + args[:image_size][1]*args[:image_size][2] + length(mrE.esns), num)
    for i in 1:length(mrE.esns)
        mrE.esns[i].x = zeros( mrE.esns[i].R_size, 1)
    end

    if args[:gpu]
        mrE.X = CuArray(mrE.X)
        for i in 1:length(mrE.esns)
            mrE.esns[i].x = CuArray(mrE.esns[i].x)
        end
    end

    __fill_X_MrESN_mnist!(mrE,args)
    __make_Rout_MrESN_mnist!(mrE,args)
end
