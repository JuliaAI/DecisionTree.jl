function _split_mse{T<:Float64, U<:Real}(labels::Vector{T}, features::Matrix{U}, nsubfeatures::Int)
    nr, nf = size(features)
    best = NO_BEST
    best_val = -Inf

    if nsubfeatures > 0
        r = randperm(nf)
        inds = r[1:nsubfeatures]
    else
        inds = 1:nf
    end

    for i in inds
        if nr > 100
            features_i = features[:,i]
            domain_i = quantile(features_i, linspace(0.01, 0.99, 99))
            labels_i = labels
        else
            ord = sortperm(features[:,i])
            features_i = features[ord,i]
            domain_i = features_i
            labels_i = labels[ord]
        end
        for thresh in domain_i[2:end]
            value = _mse_loss(labels_i, features_i, thresh)
            if value > best_val
                best_val = value
                best = (i, thresh)
            end
        end
    end
    return best
end

function build_stump{T<:Float64, U<:Real}(labels::Vector{T}, features::Matrix{U})
    S = _split_mse(labels, features, 0)
    if S == NO_BEST
        return Leaf(mean(labels), labels)
    end
    id, thresh = S
    split = features[:,id] .< thresh
    return Node(id, thresh,
                Leaf(mean(labels[split]), labels[split]),
                Leaf(mean(labels[!split]), labels[!split]))
end

function build_tree{T<:Float64, U<:Real}(labels::Vector{T}, features::Matrix{U}, maxlabels=5, nsubfeatures=0)
    if length(labels) <= maxlabels
        return Leaf(mean(labels), labels)
    end
    S = _split_mse(labels, features, nsubfeatures)
    if S == NO_BEST
        return Leaf(mean(labels), labels)
    end
    id, thresh = S
    split = features[:,id] .< thresh
    return Node(id, thresh,
                build_tree(labels[split], features[split,:], maxlabels, nsubfeatures),
                build_tree(labels[!split], features[!split,:], maxlabels, nsubfeatures))
end

function build_forest{T<:Float64, U<:Real}(labels::Vector{T}, features::Matrix{U}, nsubfeatures::Integer, ntrees::Integer, maxlabels=5, partialsampling=0.7)
    partialsampling = partialsampling > 1.0 ? 1.0 : partialsampling
    Nlabels = length(labels)
    Nsamples = _int(partialsampling * Nlabels)
    forest = @parallel (vcat) for i in 1:ntrees
        inds = rand(1:Nlabels, Nsamples)
        build_tree(labels[inds], features[inds,:], maxlabels, nsubfeatures)
    end
    return Ensemble([forest;])
end
