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

function _hist_add!(counts::Dict{T, Int}, labels::Vector{T}, region::UnitRange{Int}) where T
    for i in region
        lbl = labels[i]
        counts[lbl] = get(counts, lbl, 0) + 1
    end
    return counts
end

_hist(labels::Vector{T}, region::UnitRange{Int} = 1:lastindex(labels)) where T =
    _hist_add!(Dict{T,Int}(), labels, region)

function _weighted_error(actual::Vector, predicted::Vector, weights::Vector{T}) where T <: Real
    mismatches = actual .!= predicted
    err = sum(weights[mismatches]) / sum(weights)
    return err
end

function majority_vote(labels::Vector)
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

function confusion_matrix(actual::Vector, predicted::Vector)
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

function _nfoldCV(classifier::Symbol, labels::Vector{T}, features::Matrix{S}, args...; rng) where {S, T}
    rng = mk_rng(rng)::Random.AbstractRNG
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
    inds = Random.randperm(rng, N)
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
            model = build_tree(labels, features,
                   n_subfeatures,
                   max_depth,
                   min_samples_leaf,
                   min_samples_split,
                   min_purity_increase;
                   rng = rng)
            if pruning_purity < 1.0
                model = prune_tree(model, pruning_purity)
            end
            predictions = apply_tree(model, test_features)
        elseif classifier == :forest
            model = build_forest(
                        labels, features,
                        n_subfeatures,
                        n_trees,
                        partial_sampling,
                        max_depth,
                        min_samples_leaf,
                        min_samples_split,
                        min_purity_increase;
                        rng = rng)
            predictions = apply_forest(model, test_features)
        elseif classifier == :stumps
            model, coeffs = build_adaboost_stumps(
                train_labels, train_features, n_iterations)
            predictions = apply_adaboost_stumps(model, coeffs, test_features)
        end
        cm = confusion_matrix(test_labels, predictions)
        accuracy[i] = cm.accuracy
        println("\nFold ", i)
        println(cm)
    end
    println("\nMean Accuracy: ", mean(accuracy))
    return accuracy
end

function nfoldCV_tree(
        labels              :: Vector{T},
        features            :: Matrix{S},
        n_folds             :: Integer,
        pruning_purity      :: Float64 = 1.0,
        max_depth           :: Integer = -1,
        min_samples_leaf    :: Integer = 1,
        min_samples_split   :: Integer = 2,
        min_purity_increase :: Float64 = 0.0;
        rng                 = Random.GLOBAL_RNG) where {S, T}
    _nfoldCV(:tree, labels, features, n_folds, pruning_purity, max_depth,
                min_samples_leaf, min_samples_split, min_purity_increase; rng=rng)
end
function nfoldCV_forest(
        labels              :: Vector{T},
        features            :: Matrix{S},
        n_folds             :: Integer,
        n_subfeatures       :: Integer = -1,
        n_trees             :: Integer = 10,
        partial_sampling    :: Float64 = 0.7,
        max_depth           :: Integer = -1,
        min_samples_leaf    :: Integer = 1,
        min_samples_split   :: Integer = 2,
        min_purity_increase :: Float64 = 0.0;
        rng                 = Random.GLOBAL_RNG) where {S, T}
    _nfoldCV(:forest, labels, features, n_folds, n_subfeatures, n_trees, partial_sampling,
                max_depth, min_samples_leaf, min_samples_split, min_purity_increase; rng=rng)
end
function nfoldCV_stumps(
        labels       ::Vector{T},
        features     ::Matrix{S},
        n_folds      ::Integer,
        n_iterations ::Integer = 10;
        rng          = Random.GLOBAL_RNG) where {S, T}
    _nfoldCV(:stumps, labels, features, n_folds, n_iterations; rng=rng)
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

function _nfoldCV(regressor::Symbol, labels::Vector{T}, features::Matrix, args...; rng) where T <: Float64
    rng = mk_rng(rng)::Random.AbstractRNG
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
    inds = Random.randperm(rng, N)
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
            model = build_tree(labels, features,
                   n_subfeatures,
                   max_depth,
                   min_samples_leaf,
                   min_samples_split,
                   min_purity_increase;
                   rng = rng)
            if pruning_purity < 1.0
                model = prune_tree(model, pruning_purity)
            end
            predictions = apply_tree(model, test_features)
        elseif regressor == :forest
            model = build_forest(
                        labels, features,
                        n_subfeatures,
                        n_trees,
                        partial_sampling,
                        max_depth,
                        min_samples_leaf,
                        min_samples_split,
                        min_purity_increase;
                        rng = rng)
            predictions = apply_forest(model, test_features)
        end
        err = mean_squared_error(test_labels, predictions)
        corr = cor(test_labels, predictions)
        r2 = R2(test_labels, predictions)
        R2s[i] = r2
        println("\nFold ", i)
        println("Mean Squared Error:     ", err)
        println("Correlation Coeff:      ", corr)
        println("Coeff of Determination: ", r2)
    end
    println("\nMean Coeff of Determination: ", mean(R2s))
    return R2s
end

function nfoldCV_tree(
    labels              :: Vector{T},
    features            :: Matrix{S},
    n_folds             :: Integer,
    pruning_purity      :: Float64 = 1.0,
    max_depth           :: Integer = -1,
    min_samples_leaf    :: Integer = 5,
    min_samples_split   :: Integer = 2,
    min_purity_increase :: Float64 = 0.0;
    rng                 = Random.GLOBAL_RNG) where {S, T <: Float64}
_nfoldCV(:tree, labels, features, n_folds, pruning_purity, max_depth,
            min_samples_leaf, min_samples_split, min_purity_increase; rng=rng)
end
function nfoldCV_forest(
    labels              :: Vector{T},
    features            :: Matrix{S},
    n_folds             :: Integer,
    n_subfeatures       :: Integer = -1,
    n_trees             :: Integer = 10,
    partial_sampling    :: Float64 = 0.7,
    max_depth           :: Integer = -1,
    min_samples_leaf    :: Integer = 5,
    min_samples_split   :: Integer = 2,
    min_purity_increase :: Float64 = 0.0;
    rng                 = Random.GLOBAL_RNG) where {S, T <: Float64}
_nfoldCV(:forest, labels, features, n_folds, n_subfeatures, n_trees, partial_sampling,
            max_depth, min_samples_leaf, min_samples_split, min_purity_increase; rng=rng)
end
