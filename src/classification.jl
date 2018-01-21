# Utilities

# Returns a dict ("Label1" => 1, "Label2" => 2, "Label3" => 3, ...)
label_index(labels) = Dict([Pair(v => k) for (k, v) in enumerate(labels)])

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

function stack_function_results(row_fun::Function, X::DataFrame)
    N = size(X, 1)
    N_cols = 0
    for row in eachrow(X[1,:])
        N_cols += length(row_fun(row)) # gets the number of columns
    end
    out = Array{Float64}(N, N_cols)
    for row in eachrow(X)
        out[row.row, :] = row_fun(row)
    end
    return out
end

################################################################################

function _split(labels::AbstractVector, features::Union{Matrix, DataFrame}, nsubfeatures::Int, weights::AbstractVector, rng::AbstractRNG)
    if weights == [0]
        _split_info_gain(labels, features, nsubfeatures, rng)
    else
        _split_neg_z1_loss(labels, features, weights)
    end
end

function _split_info_gain_loop(best, best_val, inds, features, labels, N)
    for i in inds
        ord = sortperm(features[:,i])
        features_i = features[ord,i]
        labels_i = labels[ord]

        hist1 = _hist(labels_i, 1:0)
        hist2 = _hist(labels_i)
        N1 = 0
        N2 = N

        for (d, range) in UniqueRanges(features_i)
            value = _info_gain(N1, hist1, N2, hist2)
            if value > best_val
                best_val = value
                best = (i, d)
            end

            deltaN = length(range)

            _hist_shift!(hist2, hist1, labels_i, range)
            N1 += deltaN
            N2 -= deltaN
        end
    end
    return best
end

function _split_info_gain(labels::AbstractVector, features::Union{Matrix, DataFrame}, nsubfeatures::Int,
                          rng::AbstractRNG)
    nf = size(features, 2)
    N = length(labels)

    best = NO_BEST
    best_val = -Inf

    if nsubfeatures > 0
        r = randperm(rng, nf)
        inds = r[1:nsubfeatures]
    else
        inds = 1:nf
    end

    return _split_info_gain_loop(best, best_val, inds, features, labels, N)
end

function _split_neg_z1_loss(labels::AbstractVector, features::Union{Matrix, DataFrame}, weights::Vector)
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

function build_stump(labels::AbstractVector, features::Union{Matrix, DataFrame}, weights=[0];
                     rng=Base.GLOBAL_RNG)
    S = _split(labels, features, 0, weights, rng)
    if S == NO_BEST
        return Leaf(majority_vote(labels), labels)
    end
    id, thresh = S
    split = features[:,id] .< thresh
    return Node(id, thresh,
                Leaf(majority_vote(labels[split]), labels[split]),
                Leaf(majority_vote(labels[(!).(split)]), labels[(!).(split)]),
                eltype(labels))
end

function build_tree(labels::AbstractVector, features::Union{Matrix, DataFrame}, nsubfeatures::Integer=0, maxdepth::Integer=-1; rng=Base.GLOBAL_RNG)
    rng = mk_rng(rng)::AbstractRNG
    if maxdepth < -1
        error("Unexpected value for maxdepth: $(maxdepth) (expected: maxdepth >= 0, or maxdepth = -1 for infinite depth)")
    elseif maxdepth==0
        return Leaf(majority_vote(labels), labels)
    end
    S = _split(labels, features, nsubfeatures, [0], rng)
    if S == NO_BEST
        return Leaf(majority_vote(labels), labels)
    end
    id, thresh = S
    split = features[:,id] .< thresh
    labels_left = labels[split]
    labels_right = labels[(!).(split)]
    pure_left = all(labels_left .== labels_left[1])
    pure_right = all(labels_right .== labels_right[1])
    if pure_right && pure_left
        return Node(id, thresh,
                    Leaf(labels_left[1], labels_left),
                    Leaf(labels_right[1], labels_right),
                    eltype(labels))
    elseif pure_left
        return Node(id, thresh,
                    Leaf(labels_left[1], labels_left),
                    build_tree(labels_right,features[(!).(split),:], nsubfeatures,
                               max(maxdepth-1, -1); rng=rng),
                    eltype(labels))
    elseif pure_right
        return Node(id, thresh,
                    build_tree(labels_left,features[split,:], nsubfeatures,
                               max(maxdepth-1, -1); rng=rng),
                    Leaf(labels_right[1], labels_right),
                    eltype(labels))
    else
        return Node(id, thresh,
                    build_tree(labels_left,features[split,:], nsubfeatures,
                               max(maxdepth-1, -1); rng=rng),
                    build_tree(labels_right,features[(!).(split),:], nsubfeatures,
                               max(maxdepth-1, -1); rng=rng),
                    eltype(labels))
    end
end

function prune_tree(tree::LeafOrNode, purity_thresh=1.0)
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
                        _prune_run(tree.right, purity_thresh),
                        _predict_type(tree))
        end
    end
    pruned = _prune_run(tree, purity_thresh)
    while length(pruned) < length(tree)
        tree = pruned
        pruned = _prune_run(tree, purity_thresh)
    end
    return pruned
end

apply_tree(leaf::Leaf, feature::Union{AbstractVector, DataFrameRow}) = leaf.majority

function apply_tree(tree::Node, features::Union{AbstractVector, DataFrameRow})
    if tree.featval == nothing
        return apply_tree(tree.left, features)
    elseif features[tree.featid] < tree.featval
        return apply_tree(tree.left, features)
    else
        return apply_tree(tree.right, features)
    end
end

function apply_tree(tree::LeafOrNode, features::Union{Matrix, DataFrame})
    pred_type=_predict_type(tree)
    if pred_type <: Union{CategoricalString, CategoricalValue}
        predictions = CategoricalVector{pred_type}(size(features,1))
    else
        predictions = Vector{pred_type}(size(features,1))
    end
    return _apply_tree_loop(tree, features, predictions)
end

function _apply_tree_loop(tree::LeafOrNode, features::Matrix, predictions::AbstractVector)
    for i in 1:length(predictions)
        predictions[i] = apply_tree(tree, features[i, :])
    end
    return predictions
end
function _apply_tree_loop(tree::LeafOrNode, features::DataFrame, predictions::AbstractVector)
    for row in eachrow(features)
        predictions[row.row] = apply_tree(tree, row)
    end
    return predictions
end

"""    apply_tree_proba(tree, features, labels::AbstractVector)

computes P(L=label|X) for each row in `features`. It returns a `N_row x
n_labels` matrix of probabilities, each row summing up to 1.

`labels` is a vector containing the distinct labels
(eg. ["versicolor", "virginica", "setosa"]). It specifies the column ordering
of the output matrix. """
apply_tree_proba(leaf::Leaf, features::Union{AbstractVector, DataFrameRow}, labels::AbstractVector) =
    compute_probabilities(labels, leaf.values)

function apply_tree_proba(tree::Node, features::Union{AbstractVector, DataFrameRow}, labels::AbstractVector)
    if tree.featval === nothing
        return apply_tree_proba(tree.left, features, labels)
    elseif features[tree.featid] < tree.featval
        return apply_tree_proba(tree.left, features, labels)
    else
        return apply_tree_proba(tree.right, features, labels)
    end
end

apply_tree_proba(tree::LeafOrNode, features::Union{Matrix, DataFrame}, labels) =
    stack_function_results(row->apply_tree_proba(tree, row, labels), features)

function build_forest(labels::AbstractVector, features::Union{Matrix, DataFrame}, nsubfeatures::Integer, ntrees::Integer, partialsampling=0.7, maxdepth=-1; rng=Base.GLOBAL_RNG)
    rng = mk_rng(rng)::AbstractRNG
    partialsampling = partialsampling > 1.0 ? 1.0 : partialsampling
    Nlabels = length(labels)
    Nsamples = _int(partialsampling * Nlabels)
    forest = @parallel (vcat) for i in 1:ntrees
        inds = rand(rng, 1:Nlabels, Nsamples)
        build_tree(labels[inds], features[inds,:], nsubfeatures, maxdepth;
                   rng=rng)
    end
    return Ensemble([forest;])
end

function apply_forest(forest::Ensemble, features::Union{AbstractVector, DataFrameRow})
    ntrees = length(forest)
    pred_type=_predict_type(forest)
    if pred_type <: Union{CategoricalString, CategoricalValue}
        votes = CategoricalVector{pred_type}(ntrees)
    else
        votes = Vector{pred_type}(ntrees)
    end
    for i in 1:ntrees
        votes[i] = apply_tree(forest.trees[i], features)
    end
    if typeof(votes[1]) <: Float64
        return mean(votes)
    else
        return majority_vote(votes)
    end
end

function apply_forest(forest::Ensemble, features::Union{Matrix, DataFrame})
    pred_type=_predict_type(forest)
    if pred_type <: Union{CategoricalString, CategoricalValue}
        predictions = CategoricalVector{pred_type}(size(features,1))
    else
        predictions = Vector{pred_type}(size(features,1))
    end
    return _apply_forest_loop(forest, features, predictions)
end

function _apply_forest_loop(forest::Ensemble, features::Matrix, predictions::AbstractVector)
    for i in 1:length(predictions)
        predictions[i] = apply_forest(forest, features[i, :])
    end
    return predictions
end
function _apply_forest_loop(forest::Ensemble, features::DataFrame, predictions::AbstractVector)
    for row in eachrow(features)
        predictions[row.row] = apply_forest(forest, row)
    end
    return predictions
end

"""    apply_forest_proba(forest::Ensemble, features::AbstractVector, col_labels::AbstractVector)

computes P(L=label|X) for each row in `features`. It returns a `N_row x
n_labels` matrix of probabilities, each row summing up to 1.

`col_labels` is a vector containing the distinct labels
(eg. ["versicolor", "virginica", "setosa"]). It specifies the column ordering
of the output matrix. """
function apply_forest_proba(forest::Ensemble, features::Union{AbstractVector, DataFrameRow}, labels::AbstractVector)
    votes = [apply_tree(tree, features) for tree in forest.trees]
    return compute_probabilities(labels, votes)
end

apply_forest_proba(forest::Ensemble, features::Union{Matrix, DataFrame}, labels) =
    stack_function_results(row->apply_forest_proba(forest, row, labels),
                           features)

function build_adaboost_stumps(labels::AbstractVector, features::Union{Matrix, DataFrame}, niterations::Integer; rng=Base.GLOBAL_RNG)
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

function apply_adaboost_stumps(stumps::Ensemble, coeffs::Vector{Float64}, features::Union{AbstractVector, DataFrameRow})
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

function apply_adaboost_stumps(stumps::Ensemble, coeffs::Vector{Float64}, features::Union{Matrix, DataFrame})
    pred_type=_predict_type(stumps)
    if pred_type <: Union{CategoricalString, CategoricalValue}
        predictions = CategoricalVector{pred_type}(size(features,1))
    else
        predictions = Vector{pred_type}(size(features,1))
    end
    return _apply_adaboost_stumps_loop(stumps, coeffs, features, predictions)
end

function _apply_adaboost_stumps_loop(stumps::Ensemble, coeffs::Vector{Float64}, features::Matrix, predictions::AbstractVector)
    for i in 1:length(predictions)
        predictions[i] = apply_adaboost_stumps(stumps, coeffs, features[i, :])
    end
    return predictions
end
function _apply_adaboost_stumps_loop(stumps::Ensemble, coeffs::Vector{Float64}, features::DataFrame, predictions::AbstractVector)
    for row in eachrow(features)
        predictions[row.row] = apply_adaboost_stumps(stumps, coeffs, row)
    end
    return predictions
end

"""    apply_adaboost_stumps_proba(stumps::Ensemble, coeffs, features::AbstractVector, labels::AbstractVector)

computes P(L=label|X) for each row in `features`. It returns a `N_row x
n_labels` matrix of probabilities, each row summing up to 1.

`col_labels` is a vector containing the distinct labels
(eg. ["versicolor", "virginica", "setosa"]). It specifies the column ordering
of the output matrix. """
function apply_adaboost_stumps_proba(stumps::Ensemble, coeffs::Vector{Float64},
                                     features::Union{AbstractVector, DataFrameRow}, labels::AbstractVector)
    votes = [apply_tree(stump, features) for stump in stumps.trees]
    compute_probabilities(labels, votes, coeffs)
end

function apply_adaboost_stumps_proba(stumps::Ensemble, coeffs::Vector{Float64},
                                    features::Union{Matrix, DataFrame}, labels::AbstractVector)
    stack_function_results(row->apply_adaboost_stumps_proba(stumps, coeffs, row,
                                                           labels),
                           features)
end

