
#=
    Given the dimensions of an image (in pixels) it returns a reservoir to process images of this dimensions.
=#
function new_R_img(m,n; value=0.0, sigma=1.0)
    num_nodes = m*n
    R         = zeros(num_nodes, num_nodes)
    n_dict    = Dict{Tuple{Int, Int}, Tuple{Int, Array{Tuple{Int,Int}} } }()
    
    for node in 1:num_nodes
        num           = node-1
        i,j           = floor(Int,num/n) + 1 , num%n + 1
        n_dict[(i,j)] = (node, moore_neighborhood(i,j,m,n) )
    end

    if value != 0.0
        for (k,v) in n_dict
            node, neigs = v[1], map( x -> n_dict[x][1] ,v[2])
            set_edges_value!(R, node, neigs; value=value)
        end
    else
        for (k,v) in n_dict
            node, neigs = v[1], map( x -> n_dict[x][1] ,v[2])
            set_edges_rand!(R, node, neigs; sigma=sigma)
        end
    end
        
    return R    
end