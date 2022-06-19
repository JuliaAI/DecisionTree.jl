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
        labels :: AbstractVector{T}) where {S, T}

    if node.is_leaf
        return Leaf{T}(list[node.label], labels[node.region])
    else
        left = _convert(node.l, list, labels)
        right = _convert(node.r, list, labels)
        return Node{S, T}(node.feature, node.threshold, left, right)
    end
end

function update_using_impurity!(feature_importance::Vector{Float64}, node::treeclassifier.NodeMeta{S}) where S
    if !node.is_leaf
        update_using_impurity!(feature_importance, node.l)
        update_using_impurity!(feature_importance, node.r)
        feature_importance[node.feature] += node.node_impurity - node.l.node_impurity - node.r.node_impurity
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

function _build_tree(tree::treeclassifier.Tree{S, T}, labels::AbstractVector{T}, n_features, n_samples, impurity_importance::Bool) where {S, T}
    node = _convert(tree.root, tree.list, labels[tree.labels])
    if !impurity_importance
        return Root{S, T}(node, n_features, Float64[])
    else
        fi = zeros(Float64, n_features)
        update_using_impurity!(fi, tree.root)
        return Root{S, T}(node, n_features, fi ./ n_samples)
    end
end

"""
    prune_tree(tree::Union{Root, LeafOrNode}, purity_thresh=1.0, loss::Function = util.entropy)

Prune tree based on prediction accuracy of each nodes.
* `purity_thresh`: If prediction accuracy of a stump is larger than this value, the node will be pruned and become a leaf.
* `loss`: The loss function for computing node impurity. It can be either `util.entropy`, `util.gini` or other measures of impurity depending on the loss function used for building the tree. 
Impurity importances will be recalculated by substracting impurity decrease of the pruned node divided by total number of training observations when the tree has `featim` property.
The calculation of impurity decrease is as same as that described in `impurity_importance` documentation.
This function will recurse until no stumps can be pruned.
"""
function prune_tree(tree::Union{Root{S, T}, LeafOrNode{S, T}}, purity_thresh=1.0, loss::Function = util.entropy) where {S, T}
    if purity_thresh >= 1.0
        return tree
    end
    ntt = nsample(tree)
    function _prune_run_stump(tree::LeafOrNode{S, T}, purity_thresh::Real, fi::Vector{Float64} = Float64[]) where {S, T}
        all_labels = [tree.left.values; tree.right.values]
        majority = majority_vote(all_labels)
        matches = findall(all_labels .== majority)
        purity = length(matches) / length(all_labels)
        if purity >= purity_thresh
            if !isempty(fi)
                nc = votes_distribution(all_labels)
                nt = length(all_labels)
                ncl = votes_distribution(tree.left.values)
                nl = length(tree.left.values)
                ncr = votes_distribution(tree.right.values)
                nr = nt - nl
                fi[tree.featid] -= (nt * loss(nc, nt) - nl * loss(ncl, nl) - nr * loss(ncr, nr)) / ntt
            end
            return Leaf{T}(majority, all_labels)
            
        else
            return tree
        end
    end
    function _prune_run(tree::Root{S, T}, purity_thresh::Real) where {S, T}
        fi = deepcopy(tree.featim) ## recalculate feature importances
        node = _prune_run(tree.node, purity_thresh, fi)
        return Root{S, T}(node, tree.n_feat, fi)
    end
    function _prune_run(tree::LeafOrNode{S, T}, purity_thresh::Real, fi::Vector{Float64} = Float64[]) where {S, T}
        N = length(tree)
        if N == 1        ## a Leaf
            return tree
        elseif N == 2    ## a stump
            return _prune_run_stump(tree, purity_thresh, fi)
        else
            left = _prune_run(tree.left, purity_thresh, fi)
            right = _prune_run(tree.right, purity_thresh, fi)
            return Node{S, T}(tree.featid, tree.featval, left, right)
        end
    end
    pruned = _prune_run(tree, purity_thresh)
    while length(pruned) < length(tree)
        tree = pruned
        pruned = _prune_run(tree, purity_thresh)
    end
    return pruned
end

apply_tree(tree::Root{S, T}, features::AbstractVector{S}) where {S, T} = apply_tree(tree.node, features)
apply_tree(leaf::Leaf{T}, feature::AbstractVector{S}) where {S, T} = leaf.majority

function apply_tree(tree::Node{S, T}, features::AbstractVector{S}) where {S, T}
    if tree.featid == 0
        return apply_tree(tree.left, features)
    elseif features[tree.featid] < tree.featval
        return apply_tree(tree.left, features)
    else
        return apply_tree(tree.right, features)
    end
end

apply_tree(tree::Root{S, T}, features::AbstractMatrix{S}) where {S, T} = apply_tree(tree.node, features)
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

"""    apply_tree_proba(::Root, features, col_labels::AbstractVector)

computes P(L=label|X) for each row in `features`. It returns a `N_row x
n_labels` matrix of probabilities, each row summing up to 1.

`col_labels` is a vector containing the distinct labels
(eg. ["versicolor", "virginica", "setosa"]). It specifies the column ordering
of the output matrix. """
apply_tree_proba(tree::Root{S, T}, features::AbstractVector{S}, labels) where {S, T} = 
    apply_tree_proba(tree.node, features, labels)
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
apply_tree_proba(tree::Root{S, T}, features::AbstractMatrix{S}, labels) where {S, T} = 
    apply_tree_proba(tree.node, features, labels)
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
        rng                 = Random.GLOBAL_RNG,
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

    forest = impurity_importance ? Vector{Root{S, T}}(undef, n_trees) : Vector{LeafOrNode{S, T}}(undef, n_trees)

    entropy_terms = util.compute_entropy_terms(n_samples)
    loss = (ns, n) -> util.entropy(ns, n, entropy_terms)

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
                loss = loss,
                rng = rng,
                impurity_importance = impurity_importance)
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
                loss = loss,
                impurity_importance = impurity_importance)
        end
    else
        throw("rng must of be type Integer or Random.AbstractRNG")
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
        labels              :: AbstractVector{T},
        features            :: AbstractMatrix{S},
        n_iterations        :: Integer;
        rng                  = Random.GLOBAL_RNG) where {S, T}
    N = length(labels)
    weights = ones(N) / N
    stumps = Node{S, T}[]
    coeffs = Float64[]
    n_features = size(features, 2)
    for i in 1:n_iterations
        new_stump = build_stump(labels, features, weights; rng=mk_rng(rng), impurity_importance=false)
        predictions = apply_tree(new_stump, features)
        err = _weighted_error(labels, predictions, weights)
        new_coeff = 0.5 * log((1.0 + err) / (1.0 - err))
        matches = labels .== predictions
        weights[(!).(matches)] *= exp(new_coeff)
        weights[matches] *= exp(-new_coeff)
        weights /= sum(weights)
        push!(coeffs, new_coeff)
        push!(stumps, new_stump.node)
        if err < 1e-6
            break
        end
    end
    return (Ensemble{S, T}(stumps, n_features, Float64[]), coeffs)
end

apply_adaboost_stumps(trees::Tuple{<: Ensemble{S, T}, AbstractVector{Float64}}, features::AbstractVecOrMat{S}) where {S, T} = apply_adaboost_stumps(trees..., features)

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
