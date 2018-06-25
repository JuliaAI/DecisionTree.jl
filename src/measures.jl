using LinearAlgebra
using Random

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

function _neg_z1_loss(labels::Vector, weights::Vector{T}) where T <: Real
    missmatches = labels .!= majority_vote(labels)
    loss = sum(weights[missmatches])
    return -loss
end

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

function _nfoldCV(classifier::Symbol, labels, features, args...)
    nfolds = args[end]
    if nfolds < 2
        return nothing
    end
    if classifier == :tree
        pruning_purity = args[1]
    elseif classifier == :forest
        n_subfeatures = args[1]
        n_trees = args[2]
        partial_sampling = args[3]
    elseif classifier == :stumps
        n_iterations = args[1]
    end
    N = length(labels)
    ntest = _int(floor(N / nfolds))
    inds = Random.randperm(N)
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
            model = build_tree(train_labels, train_features, 0)
            if pruning_purity < 1.0
                model = prune_tree(model, pruning_purity)
            end
            predictions = apply_tree(model, test_features)
        elseif classifier == :forest
            model = build_forest(train_labels, train_features, n_subfeatures, n_trees, partial_sampling)
            predictions = apply_forest(model, test_features)
        elseif classifier == :stumps
            model, coeffs = build_adaboost_stumps(train_labels, train_features, n_iterations)
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

nfoldCV_tree(labels::Vector, features::Matrix, pruning_purity::Real, nfolds::Integer)                                           = _nfoldCV(:tree, labels, features, pruning_purity, nfolds)
nfoldCV_forest(labels::Vector, features::Matrix, n_subfeatures::Integer, n_trees::Integer, nfolds::Integer, partial_sampling=0.7)  = _nfoldCV(:forest, labels, features, n_subfeatures, n_trees, partial_sampling, nfolds)
nfoldCV_stumps(labels::Vector, features::Matrix, n_iterations::Integer, nfolds::Integer)                                         = _nfoldCV(:stumps, labels, features, n_iterations, nfolds)

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

# Pearson's Correlation Coefficient
function cor(x, y)
    @assert(length(x) == length(y))
    @assert(length(x) > 1)

    n = length(x)

    x_mean = sum(x) / n
    y_mean = sum(y) / n

    x_centered = x .- x_mean
    y_centered = y .- y_mean

    x_var = sum(x_centered .^ 2)
    y_var = sum(y_centered .^ 2)

    xy_cov = sum(x_centered .* y_centered)

    return xy_cov / sqrt(x_var * y_var)

end

function _nfoldCV(regressor::Symbol, labels::Vector{T}, features::Matrix, args...) where T <: Float64
    nfolds = args[end]
    if nfolds < 2
        return nothing
    end
    if regressor == :tree
        maxlabels = args[1]
    elseif regressor == :forest
        n_subfeatures = args[1]
        n_trees = args[2]
        maxlabels = args[3]
        partial_sampling = args[4]
    end
    N = length(labels)
    ntest = _int(floor(N / nfolds))
    inds = Random.randperm(N)
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
            model = build_tree(train_labels, train_features, maxlabels, 0)
            predictions = apply_tree(model, test_features)
        elseif regressor == :forest
            model = build_forest(train_labels, train_features, n_subfeatures, n_trees, maxlabels, partial_sampling)
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

nfoldCV_tree(labels::Vector{T}, features::Matrix, nfolds::Integer, maxlabels::Integer = 5) where T <: Float64 =
    _nfoldCV(:tree, labels, features, maxlabels, nfolds)

nfoldCV_forest(labels::Vector{T}, features::Matrix, n_subfeatures::Integer, n_trees::Integer, nfolds::Integer, maxlabels::Integer = 5, partial_sampling = 0.7) where T <: Float64 = 
    _nfoldCV(:forest, labels, features, n_subfeatures, n_trees, maxlabels, partial_sampling, nfolds)


