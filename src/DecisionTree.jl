module DecisionTree

import Base: length, convert, promote_rule, show, start, next, done

export Leaf, Node, print_tree,
       build_stump, build_tree, prune_tree, apply_tree, nfoldCV_tree,
       build_forest, apply_forest, nfoldCV_forest,
       build_adaboost_stumps, apply_adaboost_stumps, nfoldCV_stumps,
       majority_vote, ConfusionMatrix, confusion_matrix

include("measures.jl")

immutable Leaf
    majority::Any
    values::Vector
end

immutable Node
    featid::Integer
    featval::Any
    left::Union(Leaf,Node)
    right::Union(Leaf,Node)
end

convert(::Type{Node}, x::Leaf) = Node(0, nothing, x, Leaf(nothing,[nothing]))
promote_rule(::Type{Node}, ::Type{Leaf}) = Node
promote_rule(::Type{Leaf}, ::Type{Node}) = Node

function length(tree::Union(Leaf,Node))
    s = split(string(tree), "Leaf")
    return length(s) - 1
end

function print_tree(tree::Leaf, indent=0)
    matches = find(tree.values .== tree.majority)
    ratio = string(length(matches)) * "/" * string(length(tree.values))
    println("$(tree.majority) : $(ratio)")
end

function print_tree(tree::Node, indent=0)
    println("Feature $(tree.featid), Threshold $(tree.featval)")
    print("    " ^ indent * "L-> ")
    print_tree(tree.left, indent + 1)
    print("    " ^ indent * "R-> ")
    print_tree(tree.right, indent + 1)
end

const NO_BEST=(0,0)

function _split(labels::Vector, features::Matrix, nsubfeatures::Int, weights::Vector)
    if weights == [0]
        _split_info_gain(labels, features, nsubfeatures)
    else
        _split_neg_z1_loss(labels, features, weights)
    end
end

# Provide an iterator giving the unique values and corresponding ranges
# in a sorted vector
# Note: the vector is assumed to be sorted, and no checking is done!
immutable UniqueRanges
    v::AbstractVector
end

start(u::UniqueRanges) = 1
done(u::UniqueRanges, s) = done(u.v, s)
next(u::UniqueRanges, s) = (val = u.v[s]; 
                            t = searchsortedlast(u.v, val, s, length(u.v), Base.Order.Forward);
                            ((val, s:t), t+1))

function _split_info_gain(labels::Vector, features::Matrix, nsubfeatures::Int)
    nf = size(features, 2)
    N = length(labels)

    best = NO_BEST
    best_val = -Inf

    if nsubfeatures > 0
        inds = randperm(nf)[1:nsubfeatures]
    else
        inds = [1:nf]
    end

    for i in inds
        ord = sortperm(features[:,i])
        features_i = features[ord,i]
        labels_i = labels[ord]

        hist1 = _hist(labels_i, 1:0)
        hist2 = _hist(labels_i)
        N1 = 0
        N2 = N

        for (d, range) in UniqueRanges(features_i)
            value = _info_gain(N1, hist1, N2, hist2)
            if value > best_val
                best_val = value
                best = (i, d)
            end

            deltaN = length(range)

            _hist_shift!(hist2, hist1, labels_i, range)
            N1 += deltaN
            N2 -= deltaN
        end
    end
    return best
end

function _split_neg_z1_loss(labels::Vector, features::Matrix, weights::Vector)
    nf = size(features,2)
    best = NO_BEST
    best_val = -Inf
    inds = [1:nf]
    for i in 1:nf
        domain_i = sort(unique(features[:,inds[i]]))
        for d in domain_i[2:]
            cur_split = features[:,inds[i]] .< d
            value = _neg_z1_loss(labels[cur_split], weights[cur_split]) + _neg_z1_loss(labels[!cur_split], weights[!cur_split])
            if value > best_val
                best_val = value
                best = (inds[i], d)
            end
        end
    end
    return best
end

function build_stump(labels::Vector, features::Matrix, weights=[0])
    S = _split(labels, features, 0, weights)
    if S == NO_BEST
        return Leaf(majority_vote(labels), labels)
    end
    id, thresh = S
    split = features[:,id] .< thresh
    return Node(id, thresh,
                Leaf(majority_vote(labels[split]), labels[split]),
                Leaf(majority_vote(labels[!split]), labels[!split]))
end

function build_tree(labels::Vector, features::Matrix, nsubfeatures=0)
    S = _split(labels, features, nsubfeatures, [0])
    if S == NO_BEST
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

function prune_tree(tree::Union(Leaf,Node), purity_thresh=1.0)
    function _prune_run(tree::Union(Leaf,Node), purity_thresh::Real)
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

apply_tree(leaf::Leaf, feature::Vector) = leaf.majority

function apply_tree(tree::Node, features::Vector)
    if tree.featval == nothing
        return apply_tree(tree.left, features)
    elseif features[tree.featid] < tree.featval
        return apply_tree(tree.left, features)
    else
        return apply_tree(tree.right, features)
    end
end

function apply_tree(tree::Union(Leaf,Node), features::Matrix)
    N = size(features,1)
    predictions = Array(Any,N)
    for i in 1:N
        predictions[i] = apply_tree(tree, squeeze(features[i,:],1))
    end
    return predictions
end

function build_forest(labels::Vector, features::Matrix, nsubfeatures::Integer, ntrees::Integer, partialsampling=0.7)
    partialsampling = partialsampling > 1.0 ? 1.0 : partialsampling
    Nlabels = length(labels)
    Nsamples = int(partialsampling * Nlabels)
    forest = @parallel (vcat) for i in [1:ntrees]
        inds = rand(1:Nlabels, Nsamples)
        build_tree(labels[inds], features[inds,:], nsubfeatures)
    end
    return [forest]
end

function apply_forest{T<:Union(Leaf,Node)}(forest::Vector{T}, features::Vector)
    ntrees = length(forest)
    votes = Array(Any,ntrees)
    for i in 1:ntrees
        votes[i] = apply_tree(forest[i],features)
    end
    return majority_vote(votes)
end

function apply_forest{T<:Union(Leaf,Node)}(forest::Vector{T}, features::Matrix)
    N = size(features,1)
    predictions = Array(Any,N)
    for i in 1:N
        predictions[i] = apply_forest(forest, squeeze(features[i,:],1))
    end
    return predictions
end

function build_adaboost_stumps(labels::Vector, features::Matrix, niterations::Integer)
    N = length(labels)
    weights = ones(N) / N
    stumps = Node[]
    coeffs = FloatingPoint[]
    for i in 1:niterations
        new_stump = build_stump(labels, features, weights)
        predictions = apply_tree(new_stump, features)
        err = _weighted_error(labels, predictions, weights)
        new_coeff = 0.5 * log((1.0 + err) / (1.0 - err))
        matches = labels .== predictions
        weights[!matches] *= exp(new_coeff)
        weights[matches] *= exp(-new_coeff)
        weights /= sum(weights)
        push!(coeffs, new_coeff)
        push!(stumps, new_stump)
        if err < 1e-6
            break
        end
    end
    return (stumps, coeffs)
end

function apply_adaboost_stumps{T<:Union(Leaf,Node)}(stumps::Vector{T}, coeffs::Vector{FloatingPoint}, features::Vector)
    nstumps = length(stumps)
    counts = Dict()
    for i in 1:nstumps
        prediction = apply_tree(stumps[i], features)
        counts[prediction] = get(counts, prediction, 0.0) + coeffs[i]
    end
    top_prediction = stumps[1].left.majority
    top_count = -Inf
    for (k,v) in counts
        if v > top_count
            top_prediction = k
            top_count = v
        end
    end
    return top_prediction
end

function apply_adaboost_stumps{T<:Union(Leaf,Node)}(stumps::Vector{T}, coeffs::Vector{FloatingPoint}, features::Matrix)
    N = size(features,1)
    predictions = Array(Any,N)
    for i in 1:N
        predictions[i] = apply_adaboost_stumps(stumps, coeffs, squeeze(features[i,:],1))
    end
    return predictions
end

end # module

