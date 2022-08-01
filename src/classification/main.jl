# Utilities

include("tree.jl")

# Returns a dict ("Label1" => 1, "Label2" => 2, "Label3" => 3, ...)
label_index(labels) = Dict(v => k for (k, v) in enumerate(labels))

## Helper function. Counts the votes.
## Returns a vector of probabilities (eg. [0.2, 0.6, 0.2]) which is in the same
## order as get_labels(classifier) (eg. ["versicolor", "setosa", "virginica"])
function compute_probabilities(labels::AbstractVector, votes::AbstractVector, weights=1.0)
    label2ind = label_index(labels)
    counts = zeros(Float64, length(label2ind))
    for (i, label) in enumerate(votes)
        if isa(weights, Number)
            counts[label2ind[label]] += weights
        else
            counts[label2ind[label]] += weights[i]
        end
    end
    return counts / sum(counts) # normalize to get probabilities
end

# Applies `row_fun(X_row)::AbstractVector` to each row in X
# and returns a matrix containing the resulting vectors, stacked vertically
function stack_function_results(row_fun::Function, X::AbstractMatrix)
    N = size(X, 1)
    N_cols = length(row_fun(X[1, :])) # gets the number of columns
    out = Array{Float64}(undef, N, N_cols)
    for i in 1:N
        out[i, :] = row_fun(X[i, :])
    end
    return out
end


function _convert(
        node   :: treeclassifier.NodeMeta{S},
        list   :: AbstractVector{T},
        labels :: AbstractVector{T}
    ) where {S, T}

    if node.is_leaf
        featfreq = Tuple(sum(labels[node.region] .== l) for l in list)
        return Leaf{T, length(list)}(
            Tuple(list), argmax(featfreq), featfreq, length(node.region))
    else
        left = _convert(node.l, list, labels)
        right = _convert(node.r, list, labels)
        return Node{S, T, length(list)}(
            node.feature, node.threshold, left, right)
    end
end

function update_using_impurity!(
    feature_importance::Vector{Float64},
    node::treeclassifier.NodeMeta{S}
) where S
    if !node.is_leaf
        update_using_impurity!(feature_importance, node.l)
        update_using_impurity!(feature_importance, node.r)
        feature_importance[node.feature] +=
            node.node_impurity - node.l.node_impurity - node.r.node_impurity
    end
    return
end

nsample(leaf::Leaf) = length(leaf.values)
nsample(tree::Node) = nsample(tree.left) + nsample(tree.right)
nsample(tree::Root) = nsample(tree.node)

# Numbers of observations for each unique labels
function votes_distribution(labels)
    unique_labels = unique(labels)
    votes = zeros(Int, length(unique_labels))
    @simd for label in labels
        votes[findfirst(==(label), unique_labels)] += 1
    end
    votes
end

function update_pruned_impurity!(
    tree::LeafOrNode{S, T},
    feature_importance::Vector{Float64},
    ntt::Int,
    loss::Function = util.entropy
) where {S, T}
    all_labels = [tree.left.values; tree.right.values]
    nc = votes_distribution(all_labels)
    nt = length(all_labels)
    ncl = votes_distribution(tree.left.values)
    nl = length(tree.left.values)
    ncr = votes_distribution(tree.right.values)
    nr = nt - nl
    feature_importance[tree.featid] -=
        (nt * loss(nc, nt) - nl * loss(ncl, nl) - nr * loss(ncr, nr)) / ntt
end

function update_pruned_impurity!(
    tree::LeafOrNode{S, T},
    feature_importance::Vector{Float64},
    ntt::Int,
    loss::Function = mean_squared_error
) where {S, T <: Float64}
    μl = mean(tree.left.values)
    nl = length(tree.left.values)
    μr = mean(tree.right.values)
    nr = length(tree.right.values)
    nt = nl + nr
    μt = (nl * μl + nr * μr) / nt
    feature_importance[tree.featid] -= (nt * loss([tree.left.values; tree.right.values], repeat([μt], nt)) - nl * loss(tree.left.values, repeat([μl], nl)) - nr * loss(tree.right.values, repeat([μr], nr))) / ntt
end

################################################################################

function build_stump(
        labels              :: AbstractVector{T},
        features            :: AbstractMatrix{S},
        weights              = nothing;
        rng                  = Random.GLOBAL_RNG,
        impurity_importance :: Bool = true) where {S, T}

    rng = mk_rng(rng)::Random.AbstractRNG
    t = treeclassifier.fit(
        X                   = features,
        Y                   = labels,
        W                   = weights,
        loss                = treeclassifier.util.zero_one,
        max_features        = size(features, 2),
        max_depth           = 1,
        min_samples_leaf    = 1,
        min_samples_split   = 2,
        min_purity_increase = 0.0;
        rng                 = rng)

    return _build_tree(t, labels, size(features, 2), size(features, 1), impurity_importance)
end

function build_tree(
        labels              :: AbstractVector{T},
        features            :: AbstractMatrix{S},
        n_subfeatures        = 0,
        max_depth            = -1,
        min_samples_leaf     = 1,
        min_samples_split    = 2,
        min_purity_increase  = 0.0;
        loss                 = util.entropy :: Function,
        rng                  = Random.GLOBAL_RNG,
        impurity_importance :: Bool = true) where {S, T}

    if max_depth == -1
        max_depth = typemax(Int)
    end
    if n_subfeatures == 0
        n_subfeatures = size(features, 2)
    end

    rng = mk_rng(rng)::Random.AbstractRNG
    t = treeclassifier.fit(
        X                   = features,
        Y                   = labels,
        W                   = nothing,
        loss                = loss,
        max_features        = Int(n_subfeatures),
        max_depth           = Int(max_depth),
        min_samples_leaf    = Int(min_samples_leaf),
        min_samples_split   = Int(min_samples_split),
        min_purity_increase = Float64(min_purity_increase),
        rng                 = rng)

    return _build_tree(t, labels, size(features, 2), size(features, 1), impurity_importance)
end

function _build_tree(
    tree::treeclassifier.Tree{S, T},
    labels::AbstractVector{T},
    n_features,
    n_samples,
    impurity_importance::Bool
) where {S, T}
    node = _convert(tree.root, tree.list, labels[tree.labels])
    n_classes = unique(labels) |> length
    if !impurity_importance
        return Root{S, T, n_classes}(node, n_features, Float64[])
    else
        fi = zeros(Float64, n_features)
        update_using_impurity!(fi, tree.root)
        return Root{S, T, n_classes}(node, n_features, fi ./ n_samples)
    end
end

"""
    prune_tree(tree::Union{Root, LeafOrNode}, purity_thresh=1.0, loss::Function)

Prune tree based on prediction accuracy of each node.

* `purity_thresh`: If the prediction accuracy of a stump is larger than this value, the node
  will be pruned and become a leaf.

* `loss`: The loss function for computing node impurity. Available function include
  `DecisionTree.util.entropy`, `DecisionTree.util.gini` and
  `DecisionTree.mean_squared_error`. Defaults are `entropy` and `mean_squared_error` for
  classification tree and regression tree, respectively. If the tree is not a `Root`, this
  argument does not affect the result.

For a tree of type `Root`, when any of its nodes is pruned, the `featim` field will be
updated by recomputing the impurity decrease of that node divided by the total number of
training observations and subtracting the value.  The computation of impurity decrease is
based on node impurity calculated with the loss function provided as the argument
`loss`. The algorithm is as same as that described in the `impurity_importance`
documentation.

This function will recurse until no stumps can be pruned.

Warn:
    For regression trees, pruning trees based on accuracy may not be an appropriate method.
"""
function prune_tree(
    tree::Union{Root{S, T}, LeafOrNode{S, T}},
    purity_thresh=1.0,
    loss::Function = T <: Float64 ? mean_squared_error : util.entropy
) where {S, T}
    if purity_thresh >= 1.0
        return tree
    end
    ntt = nsample(tree)
    function _prune_run_stump(
        tree::LeafOrNode{S, T, N},
        purity_thresh::Real,
        fi::Vector{Float64} = Float64[]
    ) where {S, T, N}
        combined = tree.left.values .+ tree.right.values
        total = tree.left.total + tree.right.total
        majority = argmax(combined)
        purity = combined[majority] / total
        if purity >= purity_thresh
            if !isempty(fi)
                update_pruned_impurity!(tree, fi, ntt, loss)
            end
            return Leaf{T, N}(tree.left.features, majority, combined, total)
        else
            return tree
        end
    end
    function _prune_run(tree::Root{S, T, N}, purity_thresh::Real) where {S, T, N}
        fi = deepcopy(tree.featim) ## recalculate feature importances
        node = _prune_run(tree.node, purity_thresh, fi)
        return Root{S, T, N}(node, fi)
    end
    function _prune_run(
        tree::LeafOrNode{S, T, N},
        purity_thresh::Real,
        fi::Vector{Float64} = Float64[]
    ) where {S, T, N}
        L = length(tree)
        if L == 1        ## a Leaf
            return tree
        elseif L == 2    ## a stump
            return _prune_run_stump(tree, purity_thresh, fi)
        else
            return Node{S, T, N}(
                tree.featid, tree.featval,
                _prune_run(tree.left, purity_thresh),
                _prune_run(tree.right, purity_thresh))
        end
    end
    pruned = _prune_run(tree, purity_thresh)
    while length(pruned) < length(tree)
        tree = pruned
        pruned = _prune_run(tree, purity_thresh)
    end
    return pruned
end


apply_tree(leaf::Leaf, feature::AbstractVector) = leaf.features[leaf.majority]
apply_tree(
    tree::Root{S, T},
    features::AbstractVector{S}
) where {S, T} = apply_tree(tree.node, features)

function apply_tree(tree::Node{S, T}, features::AbstractVector{S}) where {S, T}
    if tree.featid == 0
        return apply_tree(tree.left, features)
    elseif features[tree.featid] < tree.featval
        return apply_tree(tree.left, features)
    else
        return apply_tree(tree.right, features)
    end
end

apply_tree(
    tree::Root{S, T},
    features::AbstractMatrix{S}
) where {S, T} = apply_tree(tree.node, features)
function apply_tree(tree::LeafOrNode{S, T}, features::AbstractMatrix{S}) where {S, T}
    N = size(features,1)
    predictions = Array{T}(undef, N)
    for i in 1:N
        predictions[i] = apply_tree(tree, features[i, :])
    end
    if T <: Float64
        return Float64.(predictions)
    else
        return predictions
    end
end

"""
    apply_tree_proba(::Root, features, col_labels::AbstractVector)

computes P(L=label|X) for each row in `features`. It returns a `N_row x
n_labels` matrix of probabilities, each row summing up to 1.

`col_labels` is a vector containing the distinct labels
(eg. ["versicolor", "virginica", "setosa"]). It specifies the column ordering
of the output matrix.
"""
apply_tree_proba(tree::Root{S, T}, features::AbstractVector{S}, labels) where {S, T} =
    apply_tree_proba(tree.node, features, labels)
apply_tree_proba(leaf::Leaf{T}, features::AbstractVector{S}, labels) where {S, T} =
    leaf.values ./ leaf.total

function apply_tree_proba(
    tree::Node{S, T},
    features::AbstractVector{S},
    labels
) where {S, T}
    if tree.featval === nothing
        return apply_tree_proba(tree.left, features, labels)
    elseif features[tree.featid] < tree.featval
        return apply_tree_proba(tree.left, features, labels)
    else
        return apply_tree_proba(tree.right, features, labels)
    end
end
function apply_tree_proba(tree::Root{S, T}, features::AbstractMatrix{S}, labels) where {S, T}
    predictions = Vector{NTuple{length(labels), Float64}}(undef, size(features, 1))
    for i in 1:size(features, 1)
        predictions[i] = apply_tree_proba(tree, view(features, i, :), labels)
    end
    reinterpret(reshape, Float64, predictions) |> transpose |> Matrix
end

function build_forest(
        labels              :: AbstractVector{T},
        features            :: AbstractMatrix{S},
        n_subfeatures       = -1,
        n_trees             = 10,
        partial_sampling    = 0.7,
        max_depth           = -1,
        min_samples_leaf    = 1,
        min_samples_split   = 2,
        min_purity_increase = 0.0;
        rng::Union{Integer,AbstractRNG} = Random.GLOBAL_RNG,
        impurity_importance :: Bool = true) where {S, T}

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

    forest = impurity_importance ?
        Vector{Root{S, T}}(undef, n_trees) :
        Vector{LeafOrNode{S, T}}(undef, n_trees)

    entropy_terms = util.compute_entropy_terms(n_samples)
    loss = (ns, n) -> util.entropy(ns, n, entropy_terms)

    if rng isa Random.AbstractRNG
        shared_seed = rand(rng, UInt)
        Threads.@threads for i in 1:n_trees
            # The Mersenne Twister (Julia's default) is not thread-safe.
            _rng = Random.seed!(copy(rng), shared_seed + i)
            inds = rand(_rng, 1:t_samples, n_samples)
            forest[i] = build_tree(
                labels[inds],
                features[inds,:],
                n_subfeatures,
                max_depth,
                min_samples_leaf,
                min_samples_split,
                min_purity_increase,
                loss = loss,
                rng = _rng,
                impurity_importance = impurity_importance)
        end
    else # each thread gets its own seeded rng
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
                loss = loss,
                impurity_importance = impurity_importance)
        end
    end

    return _build_forest(forest, size(features, 2), n_trees, impurity_importance)
end

function _build_forest(
        forest              :: Vector{<: Union{Root{S, T}, LeafOrNode{S, T}}},
        n_features          ,
        n_trees             ,
        impurity_importance :: Bool) where {S, T}

    if !impurity_importance
        return Ensemble{S, T}(forest, n_features, Float64[])
    else
        fi = zeros(Float64, n_features)
        for tree in forest
            ti = DecisionTree.impurity_importance(tree, normalize = true)
            if !isempty(ti)
                fi .+= ti
            end
        end

        forest_new = Vector{LeafOrNode{S, T}}(undef, n_trees)
        Threads.@threads for i in 1:n_trees
            forest_new[i] = forest[i].node
        end

        return Ensemble{S, T}(forest_new, n_features, fi ./ n_trees)
    end
end

function apply_forest(forest::Ensemble{S, T}, features::AbstractVector{S}) where {S, T}
    n_trees = length(forest)
    votes = Array{T}(undef, n_trees)
    for i in 1:n_trees
        votes[i] = apply_tree(forest.trees[i], features)
    end

    if T <: Float64
        return mean(votes)
    else
        return majority_vote(votes)
    end
end

function apply_forest(
        forest::Ensemble{S, T},
        features::AbstractMatrix{S};
        use_multithreading = false
    ) where {S, T}
    N = size(features,1)
    predictions = Array{T}(undef, N)
    if use_multithreading
        Threads.@threads for i in 1:N
            predictions[i] = apply_forest(forest, @view(features[i, :]))
        end
    else
        for i in 1:N
            predictions[i] = apply_forest(forest, @view(features[i, :]))
        end
    end
    return predictions
end

"""
    apply_forest_proba(forest::Ensemble, features, col_labels::AbstractVector)

computes P(L=label|X) for each row in `features`. It returns a `N_row x
n_labels` matrix of probabilities, each row summing up to 1.

`col_labels` is a vector containing the distinct labels
(eg. ["versicolor", "virginica", "setosa"]). It specifies the column ordering
of the output matrix.
"""
function apply_forest_proba(
    forest::Ensemble{S, T},
    features::AbstractVector{S},
    labels
) where {S, T}
    votes = [apply_tree(tree, features) for tree in forest.trees]
    return compute_probabilities(labels, votes)
end

apply_forest_proba(
    forest::Ensemble{S, T},
    features::AbstractMatrix{S},
    labels
) where {S, T} =
    stack_function_results(row->apply_forest_proba(forest, row, labels),
                           features)

function build_adaboost_stumps(
        labels              :: AbstractVector{T},
        features            :: AbstractMatrix{S},
        n_iterations        :: Integer;
        rng                  = Random.GLOBAL_RNG) where {S, T}
    N = length(labels)
    n_labels = length(unique(labels))
    base_coeff = log(n_labels - 1)
    thresh = 1 - 1 / n_labels
    weights = ones(N) / N
    stumps = Node{S, T}[]
    coeffs = Float64[]
    n_features = size(features, 2)
    for i in 1:n_iterations
        new_stump = build_stump(
            labels,
            features,
            weights;
            rng=mk_rng(rng),
            impurity_importance=false
        )
        predictions = apply_tree(new_stump, features)
        err = _weighted_error(labels, predictions, weights)
        if err >= thresh # should be better than random guess
            continue
        end
        # SAMME algorithm
        new_coeff = log((1.0 - err) / err) + base_coeff
        unmatches = labels .!= predictions
        weights[unmatches] *= exp(new_coeff)
        weights /= sum(weights)
        push!(coeffs, new_coeff)
        push!(stumps, new_stump.node)
        if err < 1e-6
            break
        end
    end
    return (Ensemble{S, T}(stumps, n_features, Float64[]), coeffs)
end

apply_adaboost_stumps(
    trees::Tuple{<: Ensemble{S, T}, AbstractVector{Float64}},
    features::AbstractVecOrMat{S}
) where {S, T} = apply_adaboost_stumps(trees..., features)

function apply_adaboost_stumps(
    stumps::Ensemble{S, T},
    coeffs::AbstractVector{Float64},
    features::AbstractVector{S}
) where {S, T}
    n_stumps = length(stumps)
    counts = Dict()
    for i in 1:n_stumps
        prediction = apply_tree(stumps.trees[i], features)
        counts[prediction] = get(counts, prediction, 0.0) + coeffs[i]
    end
    top_prediction = stumps.trees[1].left.majority
    top_count = -Inf
    for (k,v) in counts
        if v > top_count
            top_prediction = k
            top_count = v
        end
    end
    return top_prediction
end

function apply_adaboost_stumps(
    stumps::Ensemble{S, T},
    coeffs::AbstractVector{Float64},
    features::AbstractMatrix{S}
) where {S, T}
    n_samples = size(features, 1)
    predictions = Array{T}(undef, n_samples)
    for i in 1:n_samples
        predictions[i] = apply_adaboost_stumps(stumps, coeffs, features[i,:])
    end
    return predictions
end

"""
    apply_adaboost_stumps_proba(stumps::Ensemble, coeffs, features, labels::AbstractVector)

computes P(L=label|X) for each row in `features`. It returns a `N_row x
n_labels` matrix of probabilities, each row summing up to 1.

`col_labels` is a vector containing the distinct labels
(eg. ["versicolor", "virginica", "setosa"]). It specifies the column ordering
of the output matrix.
"""
function apply_adaboost_stumps_proba(
    stumps::Ensemble{S, T},
    coeffs::AbstractVector{Float64},
    features::AbstractVector{S},
    labels::AbstractVector{T}
) where {S, T}
    votes = [apply_tree(stump, features) for stump in stumps.trees]
    compute_probabilities(labels, votes, coeffs)
end

function apply_adaboost_stumps_proba(
    stumps::Ensemble{S, T},
    coeffs::AbstractVector{Float64},
    features::AbstractMatrix{S},
    labels::AbstractVector{T}
) where {S, T}
    stack_function_results(
        row->apply_adaboost_stumps_proba(stumps, coeffs, row, labels),
        features
    )
end
