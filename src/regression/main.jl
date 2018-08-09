include("tree.jl")

function _convert(node::treeregressor.NodeMeta{S}, labels::Array{T}) where {S, T <: Float64}
    if node.is_leaf
        return Leaf{T}(node.label, labels[node.region])
    else
        left = _convert(node.l, labels)
        right = _convert(node.r, labels)
        return Node{S, T}(node.feature, node.threshold, left, right)
    end
end

function build_stump(labels::Vector{T}, features::Matrix{S}; rng = Random.GLOBAL_RNG) where {S, T <: Float64}
    return build_tree(labels, features, 0, 1)
end

function build_tree(
        labels             :: Vector{T},
        features           :: Matrix{S},
        n_subfeatures       = 0,
        max_depth           = -1,
        min_samples_leaf    = 5,
        min_samples_split   = 2,
        min_purity_increase = 0.0;
        rng                 = Random.GLOBAL_RNG) where {S, T <: Float64}

    if max_depth == -1
        max_depth = typemax(Int)
    end
    if n_subfeatures == 0
        n_subfeatures = size(features, 2)
    end

    rng = mk_rng(rng)::Random.AbstractRNG
    t = treeregressor.fit(
        X                   = features,
        Y                   = labels,
        W                   = nothing,
        max_features        = Int(n_subfeatures),
        max_depth           = Int(max_depth),
        min_samples_leaf    = Int(min_samples_leaf),
        min_samples_split   = Int(min_samples_split),
        min_purity_increase = Float64(min_purity_increase),
        rng                 = rng)

    return _convert(t.root, labels[t.labels])
end

function build_forest(
        labels              :: Vector{T},
        features            :: Matrix{S},
        n_subfeatures       = -1,
        n_trees             = 10,
        partial_sampling    = 0.7,
        max_depth           = -1,
        min_samples_leaf    = 5,
        min_samples_split   = 2,
        min_purity_increase = 0.0;
        rng                 = Random.GLOBAL_RNG) where {S, T <: Float64}

    if n_trees < 1
        throw("the number of trees must be >= 1")
    end
    if !(0.0 < partial_sampling <= 1.0)
        throw("partial_sampling must be in the range (0,1]")
    end

    if n_subfeatures == -1
        n_features = size(features, 2)
        n_subfeatures = round(Int, sqrt(n_features))
    end

    t_samples = length(labels)
    n_samples = floor(Int, partial_sampling * t_samples)

    rngs = mk_rng(rng)::Random.AbstractRNG
    forest = Distributed.@distributed (vcat) for i in 1:n_trees
        inds = rand(rngs, 1:t_samples, n_samples)
        build_tree(
            labels[inds],
            features[inds,:],
            n_subfeatures,
            max_depth,
            min_samples_leaf,
            min_samples_split,
            min_purity_increase,
            rng = rngs)
    end

    if n_trees == 1
        return Ensemble{S, T}([forest])
    else
        return Ensemble{S, T}(forest)
    end
end


#= temporarily commenting out new prune_tree implementation
function prune_tree(tree::LeafOrNode{S, T}, purity_thresh=0.0) where {S, T <: Float64}

    function recurse(leaf :: Leaf{T}, purity_thresh :: Float64)
        tssq = 0.0
        tsum = 0.0
        for v in leaf.values
            tssq += v*v
            tsum += v
        end

        return tssq, tsum, leaf
    end

    function recurse(node :: Node{S, T}, purity_thresh :: Float64)

        lssq, lsum, l = recurse(node.left, purity_thresh)
        rssq, rsum, r = recurse(node.right, purity_thresh)

        if is_leaf(l) && is_leaf(r)
            n_samples = length(l.values) + length(r.values)
            tsum = lsum + rsum
            tssq = lssq + rssq
            tavg = tsum / n_samples
            purity = tavg * tavg - tssq / n_samples
            if purity > purity_thresh
                return tsum, tssq, Leaf{T}(tavg, [l.values; r.values])
            end

        end

        return 0.0, 0.0, Node{S, T}(node.featid, node.featval, l, r)
    end

    _, _, node = recurse(tree, purity_thresh)
    return node
end
=#
