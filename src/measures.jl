struct ConfusionMatrix
    classes::Vector
    matrix::Matrix{Int}
    accuracy::Float64
    kappa::Float64
end

function show(io::IO, cm::ConfusionMatrix)
    print(io, "Classes:  ")
    show(io, cm.classes)
    print(io, "\nMatrix:   ")
    display(cm.matrix)
    print(io, "\nAccuracy: ")
    show(io, cm.accuracy)
    print(io, "\nKappa:    ")
    show(io, cm.kappa)
end

function _hist_add!(counts::Dict{T, Int}, labels::AbstractVector{T}, region::UnitRange{Int}) where T
    for i in region
        lbl = labels[i]
        counts[lbl] = get(counts, lbl, 0) + 1
    end
    return counts
end

_hist(labels::AbstractVector{T}, region::UnitRange{Int} = 1:lastindex(labels)) where T =
    _hist_add!(Dict{T,Int}(), labels, region)

function _weighted_error(actual::AbstractVector, predicted::AbstractVector, weights::AbstractVector{T}) where T <: Real
    mismatches = actual .!= predicted
    err = sum(weights[mismatches]) / sum(weights)
    return err
end

function majority_vote(labels::AbstractVector)
    if length(labels) == 0
        return nothing
    end
    counts = _hist(labels)
    top_vote = labels[1]
    top_count = -1
    for (k,v) in counts
        if v > top_count
            top_vote = k
            top_count = v
        end
    end
    return top_vote
end

### Classification ###

function confusion_matrix(actual::AbstractVector, predicted::AbstractVector)
    @assert length(actual) == length(predicted)
    N = length(actual)
    _actual = zeros(Int,N)
    _predicted = zeros(Int,N)
    classes = sort(unique([actual; predicted]))
    N = length(classes)
    for i in 1:N
        _actual[actual .== classes[i]] .= i
        _predicted[predicted .== classes[i]] .= i
    end
    CM = zeros(Int,N,N)
    for i in zip(_actual, _predicted)
        CM[i[1],i[2]] += 1
    end
    accuracy = LinearAlgebra.tr(CM) / sum(CM)
    prob_chance = (sum(CM,dims=1) * sum(CM,dims=2))[1] / sum(CM)^2
    kappa = (accuracy - prob_chance) / (1.0 - prob_chance)
    return ConfusionMatrix(classes, CM, accuracy, kappa)
end

function accuracy(actual, predicted)
    length(actual) == length(predicted) || error(DimensionMismatch("actrual values and predicted values should have the same length."))
    sum(actual .== predicted)/length(actual)
end

function _nfoldCV(classifier::Symbol, labels::AbstractVector{T}, features::AbstractMatrix{S}, args...; verbose, rng) where {S, T}
    _rng = mk_rng(rng)::Random.AbstractRNG
    nfolds = args[1]
    if nfolds < 2
        throw("number of folds must be greater than 1")
    end
    if classifier == :tree
        pruning_purity      = args[2]
        max_depth           = args[3]
        min_samples_leaf    = args[4]
        min_samples_split   = args[5]
        min_purity_increase = args[6]
    elseif classifier == :forest
        n_subfeatures       = args[2]
        n_trees             = args[3]
        partial_sampling    = args[4]
        max_depth           = args[5]
        min_samples_leaf    = args[6]
        min_samples_split   = args[7]
        min_purity_increase = args[8]
    elseif classifier == :stumps
        n_iterations        = args[2]
    end
    N = length(labels)
    ntest = floor(Int, N / nfolds)
    inds = Random.randperm(_rng, N)
    accuracy = zeros(nfolds)
    for i in 1:nfolds
        test_inds = falses(N)
        test_inds[(i - 1) * ntest + 1 : i * ntest] .= true
        train_inds = (!).(test_inds)
        test_features = features[inds[test_inds],:]
        test_labels = labels[inds[test_inds]]
        train_features = features[inds[train_inds],:]
        train_labels = labels[inds[train_inds]]

        if classifier == :tree
            n_subfeatures = 0
            model = build_tree(train_labels, train_features,
                   n_subfeatures,
                   max_depth,
                   min_samples_leaf,
                   min_samples_split,
                   min_purity_increase;
                   rng = rng,
                   impurity_importance = false)
            if pruning_purity < 1.0
                model = prune_tree(model, pruning_purity)
            end
            predictions = apply_tree(model, test_features)
        elseif classifier == :forest
            model = build_forest(
                        train_labels, train_features,
                        n_subfeatures,
                        n_trees,
                        partial_sampling,
                        max_depth,
                        min_samples_leaf,
                        min_samples_split,
                        min_purity_increase;
                        rng = rng,
                        impurity_importance = false)
            predictions = apply_forest(model, test_features)
        elseif classifier == :stumps
            model, coeffs = build_adaboost_stumps(
                train_labels, train_features, n_iterations; rng = rng)
            predictions = apply_adaboost_stumps(model, coeffs, test_features)
        end
        cm = confusion_matrix(test_labels, predictions)
        accuracy[i] = cm.accuracy
        if verbose
            println("\nFold ", i)
            println(cm)
        end
    end
    println("\nMean Accuracy: ", mean(accuracy))
    return accuracy
end

function nfoldCV_tree(
        labels              :: AbstractVector{T},
        features            :: AbstractMatrix{S},
        n_folds             :: Integer,
        pruning_purity      :: Float64 = 1.0,
        max_depth           :: Integer = -1,
        min_samples_leaf    :: Integer = 1,
        min_samples_split   :: Integer = 2,
        min_purity_increase :: Float64 = 0.0;
        verbose             :: Bool = true,
        rng                 = Random.GLOBAL_RNG) where {S, T}
    _nfoldCV(:tree, labels, features, n_folds, pruning_purity, max_depth,
                min_samples_leaf, min_samples_split, min_purity_increase; verbose=verbose, rng=rng)
end
function nfoldCV_forest(
        labels              :: AbstractVector{T},
        features            :: AbstractMatrix{S},
        n_folds             :: Integer,
        n_subfeatures       :: Integer = -1,
        n_trees             :: Integer = 10,
        partial_sampling    :: Float64 = 0.7,
        max_depth           :: Integer = -1,
        min_samples_leaf    :: Integer = 1,
        min_samples_split   :: Integer = 2,
        min_purity_increase :: Float64 = 0.0;
        verbose             :: Bool = true,
        rng                 = Random.GLOBAL_RNG) where {S, T}
    _nfoldCV(:forest, labels, features, n_folds, n_subfeatures, n_trees, partial_sampling,
                max_depth, min_samples_leaf, min_samples_split, min_purity_increase; verbose=verbose, rng=rng)
end
function nfoldCV_stumps(
        labels       ::AbstractVector{T},
        features     ::AbstractMatrix{S},
        n_folds      ::Integer,
        n_iterations ::Integer = 10;
        verbose             :: Bool = true,
        rng          = Random.GLOBAL_RNG) where {S, T}
    rng = mk_rng(rng)::Random.AbstractRNG
    _nfoldCV(:stumps, labels, features, n_folds, n_iterations; verbose=verbose, rng=rng)
end

### Regression ###

function mean_squared_error(actual, predicted)
    @assert length(actual) == length(predicted)
    return mean((actual - predicted).^2)
end

function R2(actual, predicted)
    @assert length(actual) == length(predicted)
    ss_residual = sum((actual - predicted).^2)
    ss_total = sum((actual .- mean(actual)).^2)
    return 1.0 - ss_residual/ss_total
end

function _nfoldCV(regressor::Symbol, labels::AbstractVector{T}, features::AbstractMatrix, args...; verbose, rng) where T <: Float64
    _rng = mk_rng(rng)::Random.AbstractRNG
    nfolds = args[1]
    if nfolds < 2
        throw("number of folds must be greater than 1")
    end
    if regressor == :tree
        pruning_purity      = args[2]
        max_depth           = args[3]
        min_samples_leaf    = args[4]
        min_samples_split   = args[5]
        min_purity_increase = args[6]
    elseif regressor == :forest
        n_subfeatures       = args[2]
        n_trees             = args[3]
        partial_sampling    = args[4]
        max_depth           = args[5]
        min_samples_leaf    = args[6]
        min_samples_split   = args[7]
        min_purity_increase = args[8]
    end
    N = length(labels)
    ntest = floor(Int, N / nfolds)
    inds = Random.randperm(_rng, N)
    R2s = zeros(nfolds)
    for i in 1:nfolds
        test_inds = falses(N)
        test_inds[(i - 1) * ntest + 1 : i * ntest] .= true
        train_inds = (!).(test_inds)
        test_features = features[inds[test_inds],:]
        test_labels = labels[inds[test_inds]]
        train_features = features[inds[train_inds],:]
        train_labels = labels[inds[train_inds]]
        if regressor == :tree
            n_subfeatures = 0
            model = build_tree(train_labels, train_features,
                   n_subfeatures,
                   max_depth,
                   min_samples_leaf,
                   min_samples_split,
                   min_purity_increase;
                   rng = rng,
                   impurity_importance = false)
            if pruning_purity < 1.0
                model = prune_tree(model, pruning_purity)
            end
            predictions = apply_tree(model, test_features)
        elseif regressor == :forest
            model = build_forest(
                        train_labels, train_features,
                        n_subfeatures,
                        n_trees,
                        partial_sampling,
                        max_depth,
                        min_samples_leaf,
                        min_samples_split,
                        min_purity_increase;
                        rng = rng,
                        impurity_importance = false)
            predictions = apply_forest(model, test_features)
        end
        err = mean_squared_error(test_labels, predictions)
        corr = cor(test_labels, predictions)
        r2 = R2(test_labels, predictions)
        R2s[i] = r2
        if verbose
            println("\nFold ", i)
            println("Mean Squared Error:     ", err)
            println("Correlation Coeff:      ", corr)
            println("Coeff of Determination: ", r2)
        end
    end
    println("\nMean Coeff of Determination: ", mean(R2s))
    return R2s
end

function nfoldCV_tree(
    labels              :: AbstractVector{T},
    features            :: AbstractMatrix{S},
    n_folds             :: Integer,
    pruning_purity      :: Float64 = 1.0,
    max_depth           :: Integer = -1,
    min_samples_leaf    :: Integer = 5,
    min_samples_split   :: Integer = 2,
    min_purity_increase :: Float64 = 0.0;
    verbose             :: Bool = true,
    rng                 = Random.GLOBAL_RNG) where {S, T <: Float64}
_nfoldCV(:tree, labels, features, n_folds, pruning_purity, max_depth,
            min_samples_leaf, min_samples_split, min_purity_increase; verbose=verbose, rng=rng)
end
function nfoldCV_forest(
    labels              :: AbstractVector{T},
    features            :: AbstractMatrix{S},
    n_folds             :: Integer,
    n_subfeatures       :: Integer = -1,
    n_trees             :: Integer = 10,
    partial_sampling    :: Float64 = 0.7,
    max_depth           :: Integer = -1,
    min_samples_leaf    :: Integer = 5,
    min_samples_split   :: Integer = 2,
    min_purity_increase :: Float64 = 0.0;
    verbose             :: Bool = true,
    rng                 = Random.GLOBAL_RNG) where {S, T <: Float64}
_nfoldCV(:forest, labels, features, n_folds, n_subfeatures, n_trees, partial_sampling,
            max_depth, min_samples_leaf, min_samples_split, min_purity_increase; verbose=verbose, rng=rng)
end

#################################################################################################
# Feature importances
"""
    impurity_importance(tree; normalize::Bool = false)
    impurity_importance(forest)
    impurity_importance(adaboost, coeffs)

Return an vector of feature importance calculated by `Mean Decrease in Impurity (MDI)`.

Feature importance is computed as follows:
* Single tree: For each feature the associated importance is the sum, over all splits based on that feature, of the impurity decreases for that split (the node impurity minus the sum of the child impurities) divided by the total number of training observations. When `normalize` was true, the feature importances were normalized by the sum of feature importances. 
More explicitly, the impurity decrease for node i is:

    Δimpurityᵢ = nᵢ × lossᵢ - nₗ × lossₗ - nᵣ × lossᵣ

where n is number of observations, loss is entropy, gini index or other measures of impurity, index i denotes quantity of node i, index l denotes quantity of left child node, index r denotes quantity of right child node.
* Forests: The importance for a given feature is the average over trees in the forest of the **normalized** tree importances for that feature.
* AdaBoost models: The features importance is as same as `split_importance`.

For forests and adaboost models, feature imortance is normalized before avareging over trees, so keyword arguments `normalize` is useless.
Whether to normalize or not is controversial, but current implementation is identical to `scikitlearn`'s RandomForestClassifier, RandomForestRegressor and AdaBoostClassifier which is different from feature importances described in G. Louppe, “Understanding Random Forests: From Theory to Practice”, PhD Thesis, U. of Liege, 2014. (https://arxiv.org/abs/1407.7502).
See this [PR](https://github.com/scikit-learn/scikit-learn/issues/19972) for detailed discussion.

If `impurity_importance` was set false when building the tree, this function returns an empty vector.

Warn:
    The importance might be misleading because MDI is a biased method.
    See [Beware Default Random Forest Importances](https://explained.ai/rf-importance/index.html) for more dicussion.
"""
impurity_importance(tree::T; normalize::Bool = false) where {T <: DecisionTree.Root} = 
    (normalize && !isempty(tree.featim)) ? tree.featim ./ sum(tree.featim) : tree.featim

impurity_importance(forest::T; kwargs...) where {T <: DecisionTree.Ensemble} = forest.featim
impurity_importance(stumps::T, coeffs::Vector{Float64}; kwargs...) where {T <: DecisionTree.Ensemble} = 
    split_importance(stumps, coeffs)
impurity_importance(node::T; kwargs...) where {T <: DecisionTree.Node} = Float64[]
impurity_importance(lf::T; kwargs...) where {T <: DecisionTree.Leaf} = Float64[]

"""
    split_importance(tree; normalize::Bool = false)
    split_importance(forest)
    split_importance(adaboost, coeffs)

Return an vector of feature importance based on number of times feature was used in a split.

Feature importance is computed as follows:
* Single tree: For each feature the associated importance is the number of splits based on that feature.
* Forests: The importance for a given feature is the average over trees in the forest of the **normalized** tree importances for that feature.
* AdaBoost models: The importance of each feature is mean number of splits based on that feature across each stumps, that weighted by estimator weights (`coeffs`). 

For forests and adaboost models, feature imortance is normalized before avareging over trees, so keyword arguments `normalize` is useless.
"""
function split_importance(tree::T; normalize::Bool = false) where {T <: DecisionTree.Root}
    feature_importance = zeros(Float64, tree.n_feat)
    update_using_split!(feature_importance, tree.node)
    normalize ? feature_importance ./ sum(feature_importance) : feature_importance
end

function split_importance(forest::T; kwargs...) where {T <: DecisionTree.Ensemble}
    feature_importance = zeros(Float64, forest.n_feat)
    for tree in forest.trees
        ti = _split_importance(tree, forest.n_feat)
        if !isempty(ti)
            feature_importance .+= ti
        end
    end
    feature_importance ./ length(forest)
end

function split_importance(stumps::T, coeffs::Vector{Float64}; kwargs...) where {T <: DecisionTree.Ensemble}
    feature_importance = zeros(Float64, stumps.n_feat)
    for (coeff, stump) in zip(coeffs, stumps.trees)
        feature_importance[stump.featid] += coeff
    end
    feature_importance ./ sum(coeffs)
end

function _split_importance(tree::T, n::Int) where {T <: DecisionTree.Node}
    feature_importance = zeros(Float64, n)
    update_using_split!(feature_importance, tree)
    feature_importance ./ sum(feature_importance)
end


function update_using_split!(feature_importance::Vector{Float64}, node::T) where {T <: DecisionTree.Node}
    feature_importance[node.featid] += 1
    update_using_split!(feature_importance, node.left)
    update_using_split!(feature_importance, node.right)
    return
end
update_using_split!(feature_importance::Vector{Float64}, node::T) where {T <: DecisionTree.Leaf} = nothing

"""
    permutation_importances(
                            trees   :: U, 
                            labels  :: AbstractVector{T}, 
                            features:: AbstractVecOrMat{S},
                            score   :: Function,
                            n_iter  :: Int = 3;
                            rng     =  Random.GLOBAL_RNG
                            )

Calculate feature importance by shuffling each features.
* `trees`: a `DecisionTree.Leaf` object, `DecisionTree.Node` object, `DecisionTree.Root` object, `DecisionTree.Ensemble` object or `Tuple{DecisionTree.Ensemble, AbstractVector}` object (for adaboost moddel)
* `score`: a function to evaluating model performance with the form of `score(model, X, y)`

# Return a `NamedTuple`
* Fields 
1. `mean`: mean of feature importance of each shuffle
2. `std`: standard deviation of feature importance of each shuffle
3. `scores`: scores of each shuffles

For algorithm details, please see [Permutation feature importanc](https://scikit-learn.org/stable/modules/permutation_importance.html).
"""
function permutation_importance(
                                trees   :: U, 
                                labels  :: AbstractVector{T}, 
                                features:: AbstractVecOrMat{S},
                                score   :: Function,
                                n_iter  :: Int = 3;
                                rng     =  Random.GLOBAL_RNG
                                ) where {S, T, U <: Union{<: Ensemble{S, T}, <: Root{S, T}, <: DecisionTree.LeafOrNode{S, T}, Tuple{<: Ensemble{S, T}, AbstractVector{Float64}}}}

    base = score(trees, labels, features)
    n_feat = size(features, 2)
    scores = Matrix{Float64}(undef, n_feat, n_iter)
    rng = mk_rng(rng)::Random.AbstractRNG
    for i in 1:n_feat
        col = @view features[:, i]
        origin = copy(col)
        scores[i, :] = map(1:n_iter) do i
            shuffle!(rng, col)
            base - score(trees, labels, features)
        end
        features[:, i] = origin
    end

    (mean = reshape(mapslices(scores, dims = 2) do im
        mean(im)
    end, :), 
    std = reshape(mapslices(scores, dims = 2) do im
        std(im)
    end, :), 
    scores = scores)
end
