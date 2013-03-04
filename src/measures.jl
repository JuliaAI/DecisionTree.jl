function _set_entropy(labels::Vector)
    N = length(labels)
    counts = Dict()
    for i in labels
        counts[i] = get(counts, i, 0) + 1
    end
    entropy = 0
    for i in counts
        v = i[2]
        if v > 0
            entropy += v * log(v)
        end
    end
    entropy /= -N
    entropy += log(N)
    return entropy
end

function _info_gain(labels0::Vector, labels1::Vector)
    N0 = length(labels0)
    N1 = length(labels1)
    N = N0 + N1
    H = - N0/N * _set_entropy(labels0) - N1/N * _set_entropy(labels1)
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
    counts = Dict()
    for i in labels
        counts[i] = get(counts, i, 0) + 1
    end
    top_vote = None
    top_count = -Inf
    for i in collect(counts)
        if i[2] > top_count
            top_vote = i[1]
            top_count = i[2]
        end
    end
    return top_vote
end

function _sample{T<:Real}(labels::Vector, features::Matrix{T}, nsamples::Integer)
    inds = iceil(length(labels) * rand(nsamples)) ## with replacement
    return (labels[inds], features[inds,:])
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
    prob_chance = prob_chance[1]
    kappa = (accuracy - prob_chance) / (1.0 - prob_chance)
    println(classes)
    println(CM)
    println("Accuracy ", accuracy)
    println("Kappa    ", kappa)
end

function nfoldCV_tree{T<:Real}(labels::Vector, features::Matrix{T}, pruning_purity::Real, nfolds::Integer)
    if nfolds < 2
        return
    end
    N = length(labels)
    ntest = ifloor(N / nfolds)
    inds = randperm(N)
    for i in 1:nfolds
        test_inds = falses(N)
        test_inds[(i - 1) * ntest + 1 : i * ntest] = true
        train_inds = !test_inds
        test_features = features[inds[test_inds],:]
        test_labels = labels[inds[test_inds]]
        train_features = features[inds[train_inds],:]
        train_labels = labels[inds[train_inds]]
        model = build_tree(train_labels, train_features, 0)
        if pruning_purity < 1.0
            model = prune_tree(model, pruning_purity)
        end
        predictions = apply_tree(model, test_features)
        println()
        println("Fold ", i)
        confusion_matrix(test_labels, predictions)
    end
end

function nfoldCV_forest{T<:Real}(labels::Vector, features::Matrix{T}, nsubfeatures::Integer, ntrees::Integer, nfolds::Integer)
    if nfolds < 2 || ntrees < 1
        return
    end
    N = length(labels)
    ntest = ifloor(N / nfolds)
    inds = randperm(N)
    for i in 1:nfolds
        test_inds = falses(N)
        test_inds[(i - 1) * ntest + 1 : i * ntest] = true
        train_inds = !test_inds
        test_features = features[inds[test_inds],:]
        test_labels = labels[inds[test_inds]]
        train_features = features[inds[train_inds],:]
        train_labels = labels[inds[train_inds]]
        if ntrees == 1
            model = build_tree(train_labels, train_features, nsubfeatures)
            predictions = apply_tree(model, test_features)
        else
            model = build_forest(train_labels, train_features, nsubfeatures, ntrees)
            predictions = apply_forest(model, test_features)
        end
        println()
        println("Fold ", i)
        confusion_matrix(test_labels, predictions)
    end
end

function nfoldCV_stumps{T<:Real}(labels::Vector, features::Matrix{T}, niterations::Integer, nfolds::Integer)
    if nfolds < 2 || niterations < 1
        return
    end
    N = length(labels)
    ntest = ifloor(N / nfolds)
    inds = randperm(N)
    for i in 1:nfolds
        test_inds = falses(N)
        test_inds[(i - 1) * ntest + 1 : i * ntest] = true
        train_inds = !test_inds
        test_features = features[inds[test_inds],:]
        test_labels = labels[inds[test_inds]]
        train_features = features[inds[train_inds],:]
        train_labels = labels[inds[train_inds]]
        model, coeffs = build_adaboost_stumps(train_labels, train_features, niterations)
        predictions = apply_adaboost_stumps(model, coeffs, test_features)
        println()
        println("Fold ", i)
        confusion_matrix(test_labels, predictions)
    end
end

