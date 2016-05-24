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
        # Sorting used to be performed only when nr <= 100, but doing it
        # unconditionally improved fitting performance by 20%. It's a bit of a
        # puzzle. Either it improved type-stability, or perhaps branch
        # prediction is much better on a sorted sequence.
        ord = sortperm(features[:,i])
        features_i = features[ord,i]
        labels_i = labels[ord]
        if nr > 100
            domain_i = quantile(features_i, linspace(0.01, 0.99, 99);
                                sorted=true)
        else
            domain_i = features_i
        end
        value, thresh = _best_mse_loss(labels_i, features_i, domain_i)
        if value > best_val
            best_val = value
            best = (i, thresh)
        end
    end

    return best
end

""" Finds the threshold to split `features` with that minimizes the
mean-squared-error loss over `labels`.

Returns (best_val, best_thresh), where `best_val` is -MSE """
function _best_mse_loss{T<:Float64, U<:Real}(labels::Vector{T}, features::Vector{U}, domain)
    # True, but costly assert. However, see
    # https://github.com/JuliaStats/StatsBase.jl/issues/164
    # @assert issorted(features) && issorted(domain) 
    best_val = -Inf
    best_thresh = 0.0
    s_l = s2_l = zero(T)
    su = sum(labels)::T
    su2 = zero(T); for l in labels su2 += l*l end  # sum of squares
    nl = 0
    n = length(labels)
    i = 1
    # Because the `features` are sorted, below is an O(N) algorithm for finding
    # the optimal threshold amongst `domain`. We simply iterate through the
    # array and update s_l and s_r (= sum(labels) - s_l) as we go. - @cstjean
    @inbounds for thresh in domain[2:end]
        while i <= length(labels) && features[i] < thresh
            l = labels[i]

            s_l += l
            s2_l += l*l
            nl += 1

            i += 1
        end
        s_r = su - s_l
        s2_r = su2 - s2_l
        nr = n - nl
        # This check is necessary I think because in theory all labels could
        # be the same, then either nl or nr would be 0. - @cstjean
        if nr > 0 && nl > 0
            loss = s2_l - s_l^2/nl + s2_r - s_r^2/nr
            if -loss > best_val
                best_val = -loss
                best_thresh = thresh
            end
        end
    end
    return best_val, best_thresh
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
