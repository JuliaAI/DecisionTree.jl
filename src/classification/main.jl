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
        return Leaf{T}(list[node.label], labels[node.region])
    else
        left = _convert(node.l, list, labels)
        right = _convert(node.r, list, labels)
        return Node{S, T}(node.feature, node.threshold, left, right)
    end
end

################################################################################

function build_stump(
        labels      :: AbstractVector{T},
        features    :: AbstractMatrix{S},
        weights      = nothing;
        rng          = Random.GLOBAL_RNG) where {S, T}

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

    return _convert(t.root, t.list, labels[t.labels])
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
        rng                  = Random.GLOBAL_RNG) where {S, T}

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

    return _convert(t.root, t.list, labels[t.labels])
end

function prune_tree(tree::LeafOrNode{S, T}, purity_thresh=1.0) where {S, T}
    if purity_thresh >= 1.0
        return tree
    end
    function _prune_run(tree::LeafOrNode{S, T}, purity_thresh::Real) where {S, T}
        N = length(tree)
        if N == 1        ## a Leaf
            return tree
        elseif N == 2    ## a stump
            all_labels = [tree.left.values; tree.right.values]
            majority = majority_vote(all_labels)
            matches = findall(all_labels .== majority)
            purity = length(matches) / length(all_labels)
            if purity >= purity_thresh
                return Leaf{T}(majority, all_labels)
            else
                return tree
            end
        else
            return Node{S, T}(tree.featid, tree.featval,
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


apply_tree(leaf::Leaf, feature::AbstractVector) = leaf.majority

function apply_tree(tree::Node{S, T}, features::AbstractVector{S}) where {S, T}
    if tree.featid == 0
        return apply_tree(tree.left, features)
    elseif features[tree.featid] < tree.featval
        return apply_tree(tree.left, features)
    else
        return apply_tree(tree.right, features)
    end
end

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

"""    apply_tree_proba(::Node, features, col_labels::AbstractVector)

computes P(L=label|X) for each row in `features`. It returns a `N_row x
n_labels` matrix of probabilities, each row summing up to 1.

`col_labels` is a vector containing the distinct labels
(eg. ["versicolor", "virginica", "setosa"]). It specifies the column ordering
of the output matrix. """
apply_tree_proba(leaf::Leaf{T}, features::AbstractVector{S}, labels) where {S, T} =
    compute_probabilities(labels, leaf.values)

function apply_tree_proba(tree::Node{S, T}, features::AbstractVector{S}, labels) where {S, T}
    if tree.featval === nothing
        return apply_tree_proba(tree.left, features, labels)
    elseif features[tree.featid] < tree.featval
        return apply_tree_proba(tree.left, features, labels)
    else
        return apply_tree_proba(tree.right, features, labels)
    end
end

apply_tree_proba(tree::LeafOrNode{S, T}, features::AbstractMatrix{S}, labels) where {S, T} =
    stack_function_results(row->apply_tree_proba(tree, row, labels), features)

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
        rng::Union{Integer,AbstractRNG} = Random.GLOBAL_RNG) where {S, T}

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

    entropy_terms = util.compute_entropy_terms(n_samples)
    loss = (ns, n) -> util.entropy(ns, n, entropy_terms)

    if rng isa Random.AbstractRNG
        Threads.@threads for i in 1:n_trees
            # The Mersenne Twister (Julia's default) is not thread-safe.
            _rng = copy(rng)
            # Take some elements from the ring to have different states for each tree.
            # This is the only way given that only a `copy` can be expected to exist for RNGs.
            rand(_rng, i)
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
                rng = _rng)
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
                loss = loss)
        end
    end

    return Ensemble{S, T}(forest)
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

function apply_forest(forest::Ensemble{S, T}, features::AbstractMatrix{S}) where {S, T}
    N = size(features,1)
    predictions = Array{T}(undef, N)
    for i in 1:N
        predictions[i] = apply_forest(forest, features[i, :])
    end
    return predictions
end

"""    apply_forest_proba(forest::Ensemble, features, col_labels::AbstractVector)

computes P(L=label|X) for each row in `features`. It returns a `N_row x
n_labels` matrix of probabilities, each row summing up to 1.

`col_labels` is a vector containing the distinct labels
(eg. ["versicolor", "virginica", "setosa"]). It specifies the column ordering
of the output matrix. """
function apply_forest_proba(forest::Ensemble{S, T}, features::AbstractVector{S}, labels) where {S, T}
    votes = [apply_tree(tree, features) for tree in forest.trees]
    return compute_probabilities(labels, votes)
end

apply_forest_proba(forest::Ensemble{S, T}, features::AbstractMatrix{S}, labels) where {S, T} =
    stack_function_results(row->apply_forest_proba(forest, row, labels),
                           features)

function build_adaboost_stumps(
        labels       :: AbstractVector{T},
        features     :: AbstractMatrix{S},
        n_iterations :: Integer;
        rng = Random.GLOBAL_RNG) where {S, T}
    N = length(labels)
    n_labels = length(unique(labels))
    base_coeff = log(n_labels - 1)
    thresh = 1 - 1 / n_labels
    weights = ones(N) / N
    stumps = Node{S, T}[]
    coeffs = Float64[]
    for i in 1:n_iterations
        new_stump = build_stump(labels, features, weights; rng=rng)
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
        push!(stumps, new_stump)
        if err < 1e-6
            break
        end
    end
    return (Ensemble{S, T}(stumps), coeffs)
end

function apply_adaboost_stumps(stumps::Ensemble{S, T}, coeffs::AbstractVector{Float64}, features::AbstractVector{S}) where {S, T}
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

function apply_adaboost_stumps(stumps::Ensemble{S, T}, coeffs::AbstractVector{Float64}, features::AbstractMatrix{S}) where {S, T}
    n_samples = size(features, 1)
    predictions = Array{T}(undef, n_samples)
    for i in 1:n_samples
        predictions[i] = apply_adaboost_stumps(stumps, coeffs, features[i,:])
    end
    return predictions
end

"""    apply_adaboost_stumps_proba(stumps::Ensemble, coeffs, features, labels::AbstractVector)

computes P(L=label|X) for each row in `features`. It returns a `N_row x
n_labels` matrix of probabilities, each row summing up to 1.

`col_labels` is a vector containing the distinct labels
(eg. ["versicolor", "virginica", "setosa"]). It specifies the column ordering
of the output matrix. """
function apply_adaboost_stumps_proba(stumps::Ensemble{S, T}, coeffs::AbstractVector{Float64},
                                     features::AbstractVector{S}, labels::AbstractVector{T}) where {S, T}
    votes = [apply_tree(stump, features) for stump in stumps.trees]
    compute_probabilities(labels, votes, coeffs)
end

function apply_adaboost_stumps_proba(stumps::Ensemble{S, T}, coeffs::AbstractVector{Float64},
                                    features::AbstractMatrix{S}, labels::AbstractVector{T}) where {S, T}
    stack_function_results(row->apply_adaboost_stumps_proba(stumps, coeffs, row, labels), features)
end
