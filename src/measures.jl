type ConfusionMatrix
    classes::Vector
    matrix::Matrix{Int}
    accuracy::FloatingPoint
    kappa::FloatingPoint
end

function show(io::IO, cm::ConfusionMatrix)
    print(io, "Classes:  ")
    show(io, cm.classes)
    print(io, "\nMatrix:   ")
    show(io, cm.matrix)
    print(io, "\nAccuracy: ")
    show(io, cm.accuracy)
    print(io, "\nKappa:    ")
    show(io, cm.kappa)
end

function _hist_add!{T}(counts::Dict{T,Int}, labels::Vector{T}, region::Range1{Int})
    for i in region
        lbl = labels[i]
        counts[lbl] = get(counts, lbl, 0) + 1
    end
    return counts
end

function _hist_sub!{T}(counts::Dict{T,Int}, labels::Vector{T}, region::Range1{Int})
    for i in region
        lbl = labels[i]
        counts[lbl] -= 1
    end
    return counts
end

function _hist_shift!{T}(counts_from::Dict{T,Int}, counts_to::Dict{T,Int}, labels::Vector{T}, region::Range1{Int})
    for i in region
        lbl = labels[i]
        counts_from[lbl] -= 1
        counts_to[lbl] = get(counts_to, lbl, 0) + 1
    end
    return nothing
end

_hist{T}(labels::Vector{T}, region::Range1{Int} = 1:endof(labels)) = 
    _hist_add!(Dict{T,Int}(), labels, region)

function _set_entropy{T}(counts::Dict{T,Int}, N::Int)
    entropy = 0.0
    for v in values(counts)
        if v > 0
            entropy += v * log(v)
        end
    end
    entropy /= -N
    entropy += log(N)
    return entropy
end

_set_entropy(labels::Vector) = _set_entropy(_hist(labels), length(labels))

function _info_gain{T}(N1::Int, counts1::Dict{T,Int}, N2::Int, counts2::Dict{T,Int})
    N = N1 + N2
    H = - N1/N * _set_entropy(counts1, N1) - N2/N * _set_entropy(counts2, N2)
    return H
end

function _neg_z1_loss{T<:Real}(labels::Vector, weights::Vector{T})
    missmatches = labels .!= majority_vote(labels)
    loss = sum(weights[missmatches])
    return -loss
end

function _weighted_error{T<:Real}(actual::Vector, predicted::Vector, weights::Vector{T})
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

function confusion_matrix(actual::Vector, predicted::Vector)
    @assert length(actual) == length(predicted)
    N = length(actual)
    _actual = zeros(Int,N)
    _predicted = zeros(Int,N)
    classes = sort(unique([actual, predicted]))
    N = length(classes)
    for i in 1:N
        _actual[actual .== classes[i]] = i
        _predicted[predicted .== classes[i]] = i
    end
    CM = zeros(Int,N,N)
    for i in zip(_actual, _predicted)
        CM[i[1],i[2]] += 1
    end
    accuracy = trace(CM) / sum(CM)
    prob_chance = (sum(CM,1) * sum(CM,2))[1] / sum(CM)^2
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
        nsubfeatures = args[1]
        ntrees = args[2]
        partialsampling = args[3]
    elseif classifier == :stumps
        niterations = args[1]
    end
    N = length(labels)
    ntest = ifloor(N / nfolds)
    inds = randperm(N)
    accuracy = zeros(nfolds)
    for i in 1:nfolds
        test_inds = falses(N)
        test_inds[(i - 1) * ntest + 1 : i * ntest] = true
        train_inds = !test_inds
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
            model = build_forest(train_labels, train_features, nsubfeatures, ntrees, partialsampling)
            predictions = apply_forest(model, test_features)
        elseif classifier == :stumps
            model, coeffs = build_adaboost_stumps(train_labels, train_features, niterations)
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
nfoldCV_forest(labels::Vector, features::Matrix, nsubfeatures::Integer, ntrees::Integer, nfolds::Integer, partialsampling=0.7)  = _nfoldCV(:forest, labels, features, nsubfeatures, ntrees, partialsampling, nfolds)
nfoldCV_stumps(labels::Vector, features::Matrix, niterations::Integer, nfolds::Integer)                                         = _nfoldCV(:stumps, labels, features, niterations, nfolds)

