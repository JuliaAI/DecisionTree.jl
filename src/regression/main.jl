include("tree.jl")
import Random
import Distributed

# Convenience functions - make a Random Number Generator object
mk_rng(rng::Random.AbstractRNG) = rng
mk_rng(seed::Int) = Random.MersenneTwister(seed)

function build_stump(labels::Vector{T}, features::Matrix; rng = Random.GLOBAL_RNG) where T <: Float64
    return build_tree(labels, features, 1, 0, 1)
end

function build_tree(
        labels::Vector{T}, features::Matrix, min_samples_leaf = 5,
        n_subfeatures = 0, max_depth = -1, min_samples_split = 2,
        min_purity_increase = 0.0; rng = Random.GLOBAL_RNG) where T <: Float64
    rng = mk_rng(rng)::Random.AbstractRNG
    if max_depth < -1
        error("Unexpected value for max_depth: $(max_depth) (expected: max_depth >= 0, or max_depth = -1 for infinite depth)")
    end
    if max_depth == -1
        max_depth = typemax(Int64)
    end
    if n_subfeatures == 0
        n_subfeatures = size(features, 2)
    end

    t = treeregressor.fit(
        X                   = features,
        Y                   = labels,
        W                   = nothing,
        max_features        = n_subfeatures,
        max_depth           = max_depth,
        min_samples_leaf    = Int64(min_samples_leaf),
        min_samples_split   = Int64(min_samples_split),
        min_purity_increase = Float64(min_purity_increase),
        rng                 = rng)
    
    function _convert(node::treeregressor.NodeMeta, labels::Array)
        if node.is_leaf
            return Leaf(node.label, labels[node.region])
        else
            left = _convert(node.l, labels)
            right = _convert(node.r, labels)
            return Node(node.feature, node.threshold, left, right)
        end
    end
    return _convert(t.root, labels[t.labels])
end

function build_forest(
        labels::Vector{T}, features::Matrix, n_subfeatures = 0, n_trees = 10,
        min_samples_leaf = 5, partial_sampling = 0.7, max_depth = -1;
        rng = Random.GLOBAL_RNG) where T <: Float64
    rng = mk_rng(rng)::Random.AbstractRNG
    partial_sampling = partial_sampling > 1.0 ? 1.0 : partial_sampling
    Nlabels = length(labels)
    Nsamples = _int(partial_sampling * Nlabels)
    forest = @Distributed.distributed (vcat) for i in 1:n_trees
        inds = rand(rng, 1:Nlabels, Nsamples)
        build_tree(labels[inds], features[inds,:], min_samples_leaf, n_subfeatures, max_depth; rng=rng)
    end
    return Ensemble([forest;])
end
