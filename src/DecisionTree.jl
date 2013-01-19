module DecisionTree

import Base.length, Base.convert, Base.promote_rule

export Leaf, Node, RealStr,
       build_stump, build_tree, prune_tree, apply_tree,
       build_forest, apply_forest,
       build_adaboost_stumps, apply_adaboost_stumps,
       sample, majority_vote, confusion_matrix,
       nfoldCV_forest, nfoldCV_stumps

typealias RealStr Union(Real,String)
include("measures.jl")

type Leaf
    majority::RealStr
    values::Array
end

type Node
    featid::Integer
    featval::Real
    left::Union(Leaf,Node)
    right::Union(Leaf,Node)
end

convert(::Type{Node}, x::Leaf) = Node(1, Inf, x, Leaf(0,[0]))
promote_rule(::Type{Node}, ::Type{Leaf}) = Node
promote_rule(::Type{Leaf}, ::Type{Node}) = Node

function length(tree::Union(Leaf,Node))
    s = string(tree)
    s = split(s, "Leaf")
    return length(s) - 1
end

function _split{T<:RealStr}(labels::Vector{T}, features::Matrix{Float64}, nsubfeatures::Integer, weights::Vector{Float64})
    nf = size(features,2)
    best = None
    best_val = -Inf
    if nsubfeatures > 0
        inds = randperm(nf)[1:nsubfeatures]
        nf = nsubfeatures
    else
        inds = 1:nf
    end
    for i in 1:nf
        domain_i = sort(unique(features[:,inds[i]]))
        for d in domain_i[2:]
            cur_split = features[:,i] .< d
            if weights == [0]
                value = _info_gain(labels[cur_split], labels[!cur_split])
            else
                value = _neg_z1_loss(labels[cur_split], weights[cur_split]) + _neg_z1_loss(labels[!cur_split], weights[!cur_split])
            end
            if value > best_val
                best_val = value
                best = (i,d)
            end
        end
    end
    return best
end

function build_stump{T<:RealStr}(labels::Vector{T}, features::Matrix{Float64}, weights::Vector{Float64})
    S = _split(labels, features, 0, weights)
    if S == None
        return Leaf(majority_vote(labels), labels)
    end
    id, thresh = S
    split = features[:,id] .< thresh
    return Node(id, thresh,
                Leaf(majority_vote(labels[split]), labels[split]),
                Leaf(majority_vote(labels[!split]), labels[!split]))
end
build_stump{T<:RealStr}(labels::Vector{T}, features::Matrix{Float64}) = build_stump(labels, features, [0.])

function build_tree{T<:RealStr}(labels::Vector{T}, features::Matrix{Float64}, nsubfeatures::Integer)
    S = _split(labels, features, nsubfeatures, [0.])
    if S == None
        return Leaf(majority_vote(labels), labels)
    end
    id, thresh = S
    split = features[:,id] .< thresh
    return Node(id, thresh,
                build_tree(labels[split],features[split,:], nsubfeatures),
                build_tree(labels[!split],features[!split,:], nsubfeatures))
end
build_tree{T<:RealStr}(labels::Vector{T}, features::Matrix{Float64}) = build_tree(labels, features, 0)

function prune_tree{T<:Union(Leaf,Node)}(tree::T, purity_thresh::Float64)
    function _prune_run{T<:Union(Leaf,Node)}(tree::T, purity_thresh::Float64)
        N = length(tree)
        if N == 1        ## a Leaf
            return tree
        elseif N == 2    ## a stump
            all_labels = [tree.left.values, tree.right.values]
            majority = majority_vote(all_labels)
            mismatches = length(find(all_labels .!= majority))
            if mismatches == 0
                purity = 1.0
            else
                purity = mismatches / length(all_labels)
            end
            if purity > purity_thresh
                return Leaf(majority, all_labels)
            else
                return tree
            end
        else
            return Node(tree.featid, tree.featval,
                        prune_tree(tree.left, purity_thresh),
                        prune_tree(tree.right, purity_thresh))
        end
    end
    pruned = _prune_run(tree, purity_thresh)
    while length(pruned) < length(tree)
        tree = pruned
        return _prune_run(tree, purity_thresh)
    end
    return pruned
end
prune_tree{T<:Union(Leaf,Node)}(tree::T) = prune_tree(tree, 0.9) ## defaults to 90% purity pruning

function apply_tree{T<:Union(Leaf,Node)}(tree::T, features::Vector{Float64})
    if typeof(tree) == Leaf
        return tree.majority
    elseif features[tree.featid] < tree.featval
        return apply_tree(tree.left, features)
    else
        return apply_tree(tree.right, features)
    end
end

function apply_tree{T<:Union(Leaf,Node)}(tree::T, features::Matrix{Float64})
    N = size(features,1)
    predictions = zeros(Any,N)
    for i in 1:N
        predictions[i] = apply_tree(tree, squeeze(features[i,:]))
    end
    return convert(Array{UTF8String,1}, predictions)
end

function build_forest{T<:RealStr}(labels::Vector{T}, features::Matrix{Float64}, nsubfeatures::Integer, ntrees::Integer)
    N = int(0.7 * length(labels))
    forest = @parallel (vcat) for i in 1:ntrees
        _labels, _features = sample(labels, features, N)
        tree = build_tree(_labels, _features, nsubfeatures)
        prune_tree(tree)
    end
    return forest
end

function apply_forest{T<:Union(Leaf,Node)}(forest::Vector{T}, features::Vector{Float64})
    ntrees = length(forest)
    votes = zeros(Any,ntrees)
    for i in 1:ntrees
        votes[i] = apply_tree(forest[i],features)
    end
    return majority_vote(convert(Array{UTF8String,1}, votes))
end

function apply_forest{T<:Union(Leaf,Node)}(forest::Vector{T}, features::Matrix{Float64})
    N = size(features,1)
    predictions = zeros(Any,N)
    for i in 1:N
        predictions[i] = apply_forest(forest, squeeze(features[i,:]))
    end
    return convert(Array{UTF8String,1}, predictions)
end

function build_adaboost_stumps{T<:RealStr}(labels::Vector{T}, features::Matrix{Float64}, niterations::Integer)
    N = length(labels)
    weights = ones(N) / N
    stumps = []
    coeffs = []
    for i in 1:niterations
        new_stump = build_stump(labels, features, weights)
        predictions = apply_tree(new_stump, features)
        err = _weighted_error(labels, predictions, weights)
        new_coeff = log((1.0 - err) / err)
        mismatches = labels .!= predictions
        weights[mismatches] *= exp(new_coeff)
        weights /= sum(weights)
        coeffs = [coeffs, new_coeff]
        stumps = [stumps, new_stump]
        if err < 1e-6
            break
        end
    end
    return (stumps, coeffs)
end

function apply_adaboost_stumps{T<:Union(Leaf,Node)}(stumps::Vector{T}, coeffs::Vector{Float64}, features::Vector{Float64})
    nstumps = length(stumps)
    counts = Dict()
    for i in 1:nstumps
        prediction = apply_tree(stumps[i], features)
        if has(counts, prediction)
            counts[prediction] += coeffs[i]
        else
            counts[prediction] = coeffs[i]
        end
    end
    top_prediction = None
    top_count = -Inf
    for i in pairs(counts)
        if i[2] > top_count
            top_prediction = i[1]
            top_count = i[2]
        end
    end
    return top_prediction
end

function apply_adaboost_stumps{T<:Union(Leaf,Node)}(stumps::Vector{T}, coeffs::Vector{Float64}, features::Matrix{Float64})
    N = size(features,1)
    predictions = zeros(Any,N)
    for i in 1:N
        predictions[i] = apply_adaboost_stumps(stumps, coeffs, squeeze(features[i,:]))
    end
    return convert(Array{UTF8String,1}, predictions)
end

end # module

