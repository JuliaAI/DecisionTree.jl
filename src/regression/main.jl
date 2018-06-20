include("tree.jl")

# Convenience functions - make a Random Number Generator object
mk_rng(rng::AbstractRNG) = rng
mk_rng(seed::Int) = MersenneTwister(seed)

function build_stump{T<:Float64, U<:Real}(labels::Vector{T}, features::Matrix{U}; rng=Base.GLOBAL_RNG)
    return build_tree(labels, features, 1, 0, 1)
end

function build_tree{T<:Float64, U<:Real}(
        labels::Vector{T}, features::Matrix{U}, min_samples_leaf=5, nsubfeatures=0,
        max_depth=-1, min_samples_split=2, min_purity_increase=0.0;
        rng=Base.GLOBAL_RNG)
    rng = mk_rng(rng)::AbstractRNG
    if max_depth < -1
        error("Unexpected value for maxdepth: $(maxdepth) (expected: max_depth >= 0, or max_depth = -1 for infinite depth)")
    end
    if max_depth == -1
        max_depth = typemax(Int64)
    end
    if nsubfeatures == 0
        nsubfeatures = size(features, 2)
    end
    min_samples_leaf = Int64(min_samples_leaf)
    min_samples_split = Int64(min_samples_split)
    min_purity_increase = Float64(min_purity_increase)
    t = treeregressor.fit(
        features, labels, nsubfeatures, max_depth,
        min_samples_leaf, min_samples_split, min_purity_increase, 
        rng=rng)

    function _convert(node :: treeregressor.NodeMeta)
        if node.is_leaf
            return Leaf(node.label, node.labels)
        else
            left = _convert(node.l)
            right = _convert(node.r)
            return Node(node.feature, node.threshold, left, right)
        end
    end
    return _convert(t)
end

function build_forest{T<:Float64, U<:Real}(labels::Vector{T}, features::Matrix{U}, nsubfeatures::Integer, ntrees::Integer, min_samples_leaf=5, partialsampling=0.7, max_depth=-1; rng=Base.GLOBAL_RNG)
    rng = mk_rng(rng)::AbstractRNG
    partialsampling = partialsampling > 1.0 ? 1.0 : partialsampling
    Nlabels = length(labels)
    Nsamples = _int(partialsampling * Nlabels)
    forest = @parallel (vcat) for i in 1:ntrees
        inds = rand(rng, 1:Nlabels, Nsamples)
        build_tree(labels[inds], features[inds,:], min_samples_leaf, nsubfeatures, max_depth; rng=rng)
    end
    return Ensemble([forest;])
end
