# Utilities

include("tree.jl")

# Returns a dict ("Label1" => 1, "Label2" => 2, "Label3" => 3, ...)
label_index(labels) = Dict([Pair(v => k) for (k, v) in enumerate(labels)])

## Helper function. Counts the votes.
## Returns a vector of probabilities (eg. [0.2, 0.6, 0.2]) which is in the same
## order as get_labels(classifier) (eg. ["versicolor", "setosa", "virginica"])
function compute_probabilities(labels::Vector, votes::Vector, weights=1.0)
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

# Applies `row_fun(X_row)::Vector` to each row in X
# and returns a Matrix containing the resulting vectors, stacked vertically
function stack_function_results(row_fun::Function, X::Matrix)
    N = size(X, 1)
    N_cols = length(row_fun(X[1, :])) # gets the number of columns
    out = Array{Float64}(N, N_cols)
    for i in 1:N
        out[i, :] = row_fun(X[i, :])
    end
    return out
end

################################################################################

function _split_neg_z1_loss(labels::Vector, features::Matrix, weights::Vector)
    best = NO_BEST
    best_val = -Inf
    for i in 1:size(features,2)
        domain_i = sort(unique(features[:,i]))
        for thresh in domain_i[2:end]
            cur_split = features[:,i] .< thresh
            value = _neg_z1_loss(labels[cur_split], weights[cur_split]) + _neg_z1_loss(labels[(!).(cur_split)], weights[(!).(cur_split)])
            if value > best_val
                best_val = value
                best = (i, thresh)
            end
        end
    end
    return best
end

function build_stump(labels::Vector, features::Matrix, weights=[0];
                     rng=Base.GLOBAL_RNG)
    if weights == [0]
        return build_tree(labels, features, 0, 1)
    end
    S = _split_neg_z1_loss(labels, features, weights)
    if S == NO_BEST
        return Leaf(majority_vote(labels), labels)
    end
    id, thresh = S
    left = features[:,id] .< thresh
    l_labels = labels[left]
    r_labels = labels[(!).(left)]
    return Node(id, thresh,
                Leaf(majority_vote(l_labels), l_labels),
                Leaf(majority_vote(r_labels), r_labels))
end

function build_tree(labels::Vector, features::Matrix, n_subfeatures=0, max_depth=-1,
                    min_samples_leaf=1, min_samples_split=2, min_purity_increase=0.0; 
                    rng=Base.GLOBAL_RNG)
    rng = mk_rng(rng)::AbstractRNG
    if max_depth < -1
        error("Unexpected value for max_depth: $(max_depth) (expected: max_depth >= 0, or max_depth = -1 for infinite depth)")
    end
    if max_depth == -1
        max_depth = typemax(Int64)
    end
    if n_subfeatures == 0
        n_subfeatures = size(features, 2)
    end
    min_samples_leaf = Int64(min_samples_leaf)
    min_samples_split = Int64(min_samples_split)
    min_purity_increase = Float64(min_purity_increase)
    t = treeclassifier.fit(
        features, labels, n_subfeatures, max_depth,
        min_samples_leaf, min_samples_split, min_purity_increase, 
        rng=rng)

    function _convert(node :: treeclassifier.NodeMeta, labels :: Array)
        if node.is_leaf
            distribution = []
            for i in 1:length(node.labels)
                counts = node.labels[i]
                for _ in 1:counts
                    push!(distribution, labels[i])
                end
            end
            return Leaf(labels[node.label], distribution)
        else
            left = _convert(node.l, labels)
            right = _convert(node.r, labels)
            return Node(node.feature, node.threshold, left, right)
        end
    end
    return _convert(t.root, t.list)
end

function prune_tree(tree::LeafOrNode, purity_thresh=1.0)
    if purity_thresh >= 1.0
        return tree
    end
    function _prune_run(tree::LeafOrNode, purity_thresh::Real)
        N = length(tree)
        if N == 1        ## a Leaf
            return tree
        elseif N == 2    ## a stump
            all_labels = [tree.left.values; tree.right.values]
            majority = majority_vote(all_labels)
            matches = find(all_labels .== majority)
            purity = length(matches) / length(all_labels)
            if purity >= purity_thresh
                return Leaf(majority, all_labels)
            else
                return tree
            end
        else
            return Node(tree.featid, tree.featval,
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

apply_tree(leaf::Leaf, feature::Vector) = leaf.majority

function apply_tree(tree::Node, features::Vector)
    if tree.featval == nothing
        return apply_tree(tree.left, features)
    elseif features[tree.featid] < tree.featval
        return apply_tree(tree.left, features)
    else
        return apply_tree(tree.right, features)
    end
end

function apply_tree(tree::LeafOrNode, features::Matrix)
    N = size(features,1)
    predictions = Array{Any}(N)
    for i in 1:N
        predictions[i] = apply_tree(tree, features[i, :])
    end
    if typeof(predictions[1]) <: Float64
        return Float64.(predictions)
    else
        return predictions
    end
end

"""    apply_tree_proba(::Node, features, col_labels::Vector)

computes P(L=label|X) for each row in `features`. It returns a `N_row x
n_labels` matrix of probabilities, each row summing up to 1.

`col_labels` is a vector containing the distinct labels
(eg. ["versicolor", "virginica", "setosa"]). It specifies the column ordering
of the output matrix. """
apply_tree_proba(leaf::Leaf, features::Vector, labels) =
    compute_probabilities(labels, leaf.values)

function apply_tree_proba(tree::Node, features::Vector, labels)
    if tree.featval === nothing
        return apply_tree_proba(tree.left, features, labels)
    elseif features[tree.featid] < tree.featval
        return apply_tree_proba(tree.left, features, labels)
    else
        return apply_tree_proba(tree.right, features, labels)
    end
end

apply_tree_proba(tree::LeafOrNode, features::Matrix, labels) =
    stack_function_results(row->apply_tree_proba(tree, row, labels), features)

function build_forest(labels::Vector, features::Matrix, n_subfeatures=0, n_trees=10, partialsampling=0.7, max_depth=-1; rng=Base.GLOBAL_RNG)
    rng = mk_rng(rng)::AbstractRNG
    partialsampling = partialsampling > 1.0 ? 1.0 : partialsampling
    Nlabels = length(labels)
    Nsamples = _int(partialsampling * Nlabels)
    forest = @parallel (vcat) for i in 1:n_trees
        inds = rand(rng, 1:Nlabels, Nsamples)
        build_tree(labels[inds], features[inds,:], n_subfeatures, max_depth;
                   rng=rng)
    end
    return Ensemble([forest;])
end

function apply_forest(forest::Ensemble, features::Vector)
    n_trees = length(forest)
    votes = Array{Any}(n_trees)
    for i in 1:n_trees
        votes[i] = apply_tree(forest.trees[i], features)
    end
    if typeof(votes[1]) <: Float64
        return mean(votes)
    else
        return majority_vote(votes)
    end
end

function apply_forest(forest::Ensemble, features::Matrix)
    N = size(features,1)
    predictions = Array{Any}(N)
    for i in 1:N
        predictions[i] = apply_forest(forest, features[i, :])
    end
    if typeof(predictions[1]) <: Float64
        return Float64.(predictions)
    else
        return predictions
    end
end

"""    apply_forest_proba(forest::Ensemble, features, col_labels::Vector)

computes P(L=label|X) for each row in `features`. It returns a `N_row x
n_labels` matrix of probabilities, each row summing up to 1.

`col_labels` is a vector containing the distinct labels
(eg. ["versicolor", "virginica", "setosa"]). It specifies the column ordering
of the output matrix. """
function apply_forest_proba(forest::Ensemble, features::Vector, labels)
    votes = [apply_tree(tree, features) for tree in forest.trees]
    return compute_probabilities(labels, votes)
end

apply_forest_proba(forest::Ensemble, features::Matrix, labels) =
    stack_function_results(row->apply_forest_proba(forest, row, labels),
                           features)

function build_adaboost_stumps(labels::Vector, features::Matrix, niterations::Integer; rng=Base.GLOBAL_RNG)
    N = length(labels)
    weights = ones(N) / N
    stumps = Node[]
    coeffs = Float64[]
    for i in 1:niterations
        new_stump = build_stump(labels, features, weights; rng=rng)
        predictions = apply_tree(new_stump, features)
        err = _weighted_error(labels, predictions, weights)
        new_coeff = 0.5 * log((1.0 + err) / (1.0 - err))
        matches = labels .== predictions
        weights[(!).(matches)] *= exp(new_coeff)
        weights[matches] *= exp(-new_coeff)
        weights /= sum(weights)
        push!(coeffs, new_coeff)
        push!(stumps, new_stump)
        if err < 1e-6
            break
        end
    end
    return (Ensemble(stumps), coeffs)
end

function apply_adaboost_stumps(stumps::Ensemble, coeffs::Vector{Float64}, features::Vector)
    nstumps = length(stumps)
    counts = Dict()
    for i in 1:nstumps
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

function apply_adaboost_stumps(stumps::Ensemble, coeffs::Vector{Float64}, features::Matrix)
    N = size(features,1)
    predictions = Array{Any}(N)
    for i in 1:N
        predictions[i] = apply_adaboost_stumps(stumps, coeffs, features[i,:])
    end
    return predictions
end

"""    apply_adaboost_stumps_proba(stumps::Ensemble, coeffs, features, labels::Vector)

computes P(L=label|X) for each row in `features`. It returns a `N_row x
n_labels` matrix of probabilities, each row summing up to 1.

`col_labels` is a vector containing the distinct labels
(eg. ["versicolor", "virginica", "setosa"]). It specifies the column ordering
of the output matrix. """
function apply_adaboost_stumps_proba(stumps::Ensemble, coeffs::Vector{Float64},
                                     features::Vector, labels::Vector)
    votes = [apply_tree(stump, features) for stump in stumps.trees]
    compute_probabilities(labels, votes, coeffs)
end

function apply_adaboost_stumps_proba(stumps::Ensemble, coeffs::Vector{Float64},
                                    features::Matrix, labels::Vector)
    stack_function_results(row->apply_adaboost_stumps_proba(stumps, coeffs, row,
                                                           labels),
                           features)
end

