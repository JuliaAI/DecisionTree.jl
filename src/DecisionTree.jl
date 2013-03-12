module DecisionTree

import Base.length, Base.convert, Base.promote_rule, Base.show, Base.zeros

export Leaf, Node, print_tree,
       build_stump, build_tree, prune_tree, apply_tree, nfoldCV_tree,
       build_forest, apply_forest, nfoldCV_forest,
       build_adaboost_stumps, apply_adaboost_stumps, nfoldCV_stumps,
       majority_vote, ConfusionMatrix, confusion_matrix

include("measures.jl")

type Leaf
    majority::Any
    values::Vector
end

type Node
    featid::Integer
    featval::Real
    left::Union(Leaf,Node)
    right::Union(Leaf,Node)
end

zeros{T<:ByteString}(::Type{T},args...) = fill("",args...)

convert(::Type{Node}, x::Leaf) = Node(1, Inf, x, Leaf(Nothing,[Nothing]))
promote_rule(::Type{Node}, ::Type{Leaf}) = Node
promote_rule(::Type{Leaf}, ::Type{Node}) = Node

function length(tree::Union(Leaf,Node))
    s = string(tree)
    s = split(s, "Leaf")
    return length(s) - 1
end

function print_tree(tree::Union(Leaf,Node), indent::Integer)
    if typeof(tree) == Leaf
        matches = find(tree.values .== tree.majority)
        ratio = string(length(matches)) * "/" * string(length(tree.values))
        println("$(tree.majority) : $(ratio)")
    else
        println("Feature $(tree.featid), Threshold $(tree.featval)")
        print("    " ^ indent * "L-> ")
        print_tree(tree.left, indent + 1)
        print("    " ^ indent * "R-> ")
        print_tree(tree.right, indent + 1)
    end
end
print_tree(tree::Union(Leaf,Node)) = print_tree(tree, 0)

function _split{T<:Real, U<:Real}(labels::Vector, features::Matrix{T}, nsubfeatures::Integer, weights::Vector{U})
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

function build_stump{T<:Real, U<:Real}(labels::Vector, features::Matrix{T}, weights::Vector{U})
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
build_stump{T<:Real}(labels::Vector, features::Matrix{T}) = build_stump(labels, features, [0])

function build_tree{T<:Real}(labels::Vector, features::Matrix{T}, nsubfeatures::Integer)
    S = _split(labels, features, nsubfeatures, [0])
    if S == None
        return Leaf(majority_vote(labels), labels)
    end
    id, thresh = S
    split = features[:,id] .< thresh
    labels_left = labels[split]
    labels_right = labels[!split]
    pure_left = all(labels_left .== labels_left[1])
    pure_right = all(labels_right .== labels_right[1])
    if pure_right && pure_left
        return Node(id, thresh,
                    Leaf(labels_left[1], labels_left),
                    Leaf(labels_right[1], labels_right))
    elseif pure_left
        return Node(id, thresh,
                    Leaf(labels_left[1], labels_left),
                    build_tree(labels_right,features[!split,:], nsubfeatures))
    elseif pure_right
        return Node(id, thresh,
                    build_tree(labels_left,features[split,:], nsubfeatures),
                    Leaf(labels_right[1], labels_right))
    else
        return Node(id, thresh,
                    build_tree(labels_left,features[split,:], nsubfeatures),
                    build_tree(labels_right,features[!split,:], nsubfeatures))
    end
end
build_tree{T<:Real}(labels::Vector, features::Matrix{T}) = build_tree(labels, features, 0)

function prune_tree{T<:Union(Leaf,Node)}(tree::T, purity_thresh::Real)
    function _prune_run{T<:Union(Leaf,Node)}(tree::T, purity_thresh::Real)
        N = length(tree)
        if N == 1        ## a Leaf
            return tree
        elseif N == 2    ## a stump
            all_labels = [tree.left.values, tree.right.values]
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
                        _prune_run(tree.right, purity_thresh))
        end
    end
    pruned = _prune_run(tree, purity_thresh)
    while length(pruned) < length(tree)
        tree = pruned
        pruned = _prune_run(tree, purity_thresh)
    end
    return pruned
end
prune_tree{T<:Union(Leaf,Node)}(tree::T) = prune_tree(tree, 1.0) ## defaults to 100% purity pruning

function apply_tree{T<:Union(Leaf,Node), U<:Real}(tree::T, features::Vector{U})
    if typeof(tree) == Leaf
        return tree.majority
    elseif features[tree.featid] < tree.featval
        return apply_tree(tree.left, features)
    else
        return apply_tree(tree.right, features)
    end
end

function apply_tree{T<:Union(Leaf,Node), U<:Real}(tree::T, features::Matrix{U})
    N = size(features,1)
    label = apply_tree(tree, squeeze(features[1,:],1))
    predictions = zeros(typeof(label),N)
    predictions[1] = label
    for i in 2:N
        predictions[i] = apply_tree(tree, squeeze(features[i,:],1))
    end
    return predictions
end

function build_forest{T<:Real}(labels::Vector, features::Matrix{T}, nsubfeatures::Integer, ntrees::Integer)
    N = int(0.7 * length(labels))
    forest = @parallel (vcat) for i in 1:ntrees
        _labels, _features = _sample(labels, features, N)
        build_tree(_labels, _features, nsubfeatures)
    end
    return forest
end

function apply_forest{T<:Union(Leaf,Node), U<:Real}(forest::Vector{T}, features::Vector{U})
    ntrees = length(forest)
    votes = zeros(Any,ntrees)
    for i in 1:ntrees
        votes[i] = apply_tree(forest[i],features)
    end
    return majority_vote(votes)
end

function apply_forest{T<:Union(Leaf,Node), U<:Real}(forest::Vector{T}, features::Matrix{U})
    N = size(features,1)
    label = apply_forest(forest, squeeze(features[1,:],1))
    predictions = zeros(typeof(label),N)
    predictions[1] = label
    for i in 2:N
        predictions[i] = apply_forest(forest, squeeze(features[i,:],1))
    end
    return predictions
end

function build_adaboost_stumps{T<:Real}(labels::Vector, features::Matrix{T}, niterations::Integer)
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

function apply_adaboost_stumps{T<:Union(Leaf,Node), U<:Real, V<:Real}(stumps::Vector{T}, coeffs::Vector{U}, features::Vector{V})
    nstumps = length(stumps)
    counts = Dict()
    for i in 1:nstumps
        prediction = apply_tree(stumps[i], features)
        counts[prediction] = get(counts, prediction, 0.0) + coeffs[i]
    end
    top_prediction = None
    top_count = -Inf
    for i in collect(counts)
        if i[2] > top_count
            top_prediction = i[1]
            top_count = i[2]
        end
    end
    return top_prediction
end

function apply_adaboost_stumps{T<:Union(Leaf,Node), U<:Real, V<:Real}(stumps::Vector{T}, coeffs::Vector{U}, features::Matrix{V})
    N = size(features,1)
    label = apply_adaboost_stumps(stumps, coeffs, squeeze(features[1,:],1))
    predictions = zeros(typeof(label),N)
    predictions[1] = label
    for i in 2:N
        predictions[i] = apply_adaboost_stumps(stumps, coeffs, squeeze(features[i,:],1))
    end
    return predictions
end

end # module

