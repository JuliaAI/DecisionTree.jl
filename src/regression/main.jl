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

function get_ni!(feature_importance::Vector{Float64}, tree)
    if !tree.is_leaf
        get_ni!(feature_importance, tree.l)
        get_ni!(feature_importance, tree.r)
        feature_importance[tree.feature] = tree.ni - tree.l.ni - tree.r.ni
    end
end

function build_stump(labels::AbstractVector{T}, features::AbstractMatrix{S}; rng = Random.GLOBAL_RNG) where {S, T <: Float64}
    return build_tree(labels, features, 0, 1; rng=rng, calc_fi=false)
end

function build_tree(
        labels             :: AbstractVector{T},
        features           :: AbstractMatrix{S},
        n_subfeatures       = 0,
        max_depth           = -1,
        min_samples_leaf    = 5,
        min_samples_split   = 2,
        min_purity_increase = 0.0;
        rng                 = Random.GLOBAL_RNG,
        calc_fi            :: Bool = true) where {S, T <: Float64}

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
    if !calc_fi
        return _convert(t.root, labels[t.labels])
    elseif t.root.is_leaf
        return Leaf{T}(t.root.label, labels[t.root.region])
    else
        fi = zeros(Float64, size(features, 2))
        left = _convert(t.root.l, labels)
        right = _convert(t.root.r, labels)
        get_ni!(fi, t.root)
        return RootNode{S, T}(t.root.feature, t.root.threshold, left, right, fi ./ size(features, 2))
    end
end

function build_forest(
        labels              :: AbstractVector{T},
        features            :: AbstractMatrix{S},
        n_subfeatures       = -1,
        n_trees             = 10,
        partial_sampling    = 0.7,
        max_depth           = -1,
        min_samples_leaf    = 5,
        min_samples_split   = 2,
        min_purity_increase = 0.0;
        rng                 = Random.GLOBAL_RNG,
        calc_fi             :: Bool = true) where {S, T <: Float64}

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

    forest = Vector{LeafOrNode{S, T}}(undef, n_trees)

    if rng isa Random.AbstractRNG
        Threads.@threads for i in 1:n_trees
            inds = rand(rng, 1:t_samples, n_samples)
            forest[i] = build_tree(
                labels[inds],
                features[inds,:],
                n_subfeatures,
                max_depth,
                min_samples_leaf,
                min_samples_split,
                min_purity_increase,
                rng = rng,
                calc_fi = calc_fi)
        end
    elseif rng isa Integer # each thread gets its own seeded rng
        Threads.@threads for i in 1:n_trees
            Random.seed!(rng + i)
            inds = rand(1:t_samples, n_samples)
            forest[i] = build_tree(
                labels[inds],
                features[inds,:],
                n_subfeatures,
                max_depth,
                min_samples_leaf,
                min_samples_split,
                min_purity_increase,
                calc_fi = calc_fi)
        end
    else
        throw("rng must of be type Integer or Random.AbstractRNG")
    end

    if !calc_fi
        return Ensemble{S, T}(forest)
    else
        fi = zeros(Float64, size(features, 2))
        for tree in forest
            ti = feature_importances(tree, normalize = true)
            if !isempty(ti)
                fi .+= ti
            end
        end

        forest_new = Vector{LeafOrNode{S, T}}(undef, n_trees)
        Threads.@threads for i in 1:n_trees
            forest_new[i] = _convert_root(forest[i])
        end

        return Ensemble{S, T}(forest_new, fi ./ n_trees)
    end
end
