function _set_entropy{T<:RealStr}(labels::Vector{T})
    N = length(labels)
    counts = Dict()
    for i in labels
        if !has(counts,i)
            counts[i] = 0
        end
        counts[i] += 1
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

function _info_gain{T<:RealStr}(labels0::Vector{T}, labels1::Vector{T})
    N0 = length(labels0)
    N1 = length(labels1)
    N = N0 + N1
    H = - N0/N * _set_entropy(labels0) - N1/N * _set_entropy(labels1)
    return H
end

function _neg_z1_loss{T<:RealStr}(labels::Vector{T}, weights::Vector{Float64})
    missmatches = labels .!= majority_vote(labels)
    loss = sum(weights[missmatches])
    return -loss
end

function _weighted_error{T<:RealStr}(actual::Vector{T}, predicted::Vector{T}, weights::Vector{Float64})
    mismatches = actual .!= predicted
    err = sum(weights[mismatches]) / sum(weights)
    return err
end

function majority_vote{T<:RealStr}(labels::Vector{T})
    counts = Dict()
    for i in labels
        if has(counts,i)
            counts[i] += 1
        else
            counts[i] = 0
        end
    end
    top_vote = None
    top_count = -Inf
    for i in pairs(counts)
        if i[2] > top_count
            top_vote = i[1]
            top_count = i[2]
        end
    end
    return top_vote
end

function sample{T<:RealStr}(features::Matrix{Float64}, labels::Vector{T}, nsamples::Integer)
    inds = iceil(length(labels) * rand(nsamples)) ## with replacement
    return (features[inds,:], labels[inds])
end

function confusion_matrix{T<:RealStr}(actual::Vector{T}, predicted::Vector{T})
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
    kappa = (accuracy - prob_chance) / (1 - prob_chance)
    println(classes)
    println(CM)
    println("Accuracy ", accuracy)
    println("Kappa    ", kappa)
end

function nfoldCV_forest{T<:RealStr}(features::Matrix{Float64}, labels::Vector{T}, nsubfeatures::Integer, ntrees::Integer, nfolds::Integer)
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
            model = build_tree(train_features, train_labels, nsubfeatures)
            predictions = apply_tree(model, test_features)
        else
            model = build_forest(train_features, train_labels, nsubfeatures, ntrees)
            predictions = apply_forest(model, test_features)
        end
        println()
        println("Fold ", i)
        confusion_matrix(test_labels, predictions)
    end
end

function nfoldCV_stumps{T<:RealStr}(features::Matrix{Float64}, labels::Vector{T}, niterations::Integer, nfolds::Integer)
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
        model, coeffs = build_adaboost_stumps(train_features, train_labels, niterations)
        predictions = apply_adaboost_stumps(model, coeffs, test_features)
        println()
        println("Fold ", i)
        confusion_matrix(test_labels, predictions)
    end
end

