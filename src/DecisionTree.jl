__precompile__()

module DecisionTree

import Base: length, convert, promote_rule, show, start, next, done
using DataFrames

export Leaf, Node, Ensemble, print_tree, depth, build_stump, build_tree,
       prune_tree, apply_tree, apply_tree_proba, nfoldCV_tree, build_forest,
       apply_forest, apply_forest_proba, nfoldCV_forest, build_adaboost_stumps,
       apply_adaboost_stumps, apply_adaboost_stumps_proba, nfoldCV_stumps,
       majority_vote, ConfusionMatrix, confusion_matrix, mean_squared_error,
       R2, _int

# ScikitLearn API
export DecisionTreeClassifier, DecisionTreeRegressor, RandomForestClassifier,
       RandomForestRegressor, AdaBoostStumpClassifier,
       predict, predict_proba, fit!, get_classes

#####################################
##### Compatilibity Corrections #####

_int(x) = map(y->round(Integer, y), x)

###########################
########## Types ##########

const NO_BEST=(0,0)

struct Leaf{T, A <: AbstractVector{T}}
    majority::T
    values::A
end

struct Node
    featid::Int
    featval::Any
    left::Union{Leaf,Node}
    right::Union{Leaf,Node}
    predict_type::Type
end

const LeafOrNode = Union{Leaf,Node}

struct Ensemble
    trees::Vector{Node}
end

convert(::Type{Node}, x::Leaf) = Node(0, nothing, x, Leaf(nothing,[nothing]))
promote_rule(::Type{Node}, ::Type{Leaf}) = Node
promote_rule(::Type{Leaf}, ::Type{Node}) = Node

_predict_type(leaf::Leaf) = typeof(leaf.majority)
_predict_type(node::Node) = node.predict_type
_predict_type(ensemble::Ensemble) = _predict_type(first(ensemble.trees))

struct UniqueRanges{V<:AbstractVector}
    v::V
end

start(u::UniqueRanges) = 1
done(u::UniqueRanges, s) = done(u.v, s)
next(u::UniqueRanges, s) = (val = u.v[s];
                            t = searchsortedlast(u.v, val, s, length(u.v), Base.Order.Forward);
                            ((val, s:t), t+1))

##############################
########## Includes ##########

include("measures.jl")
include("classification.jl")
include("regression.jl")
include("dataframesAPI.jl")
include("scikitlearnAPI.jl")


#############################
########## Methods ##########

length(leaf::Leaf) = 1
length(tree::Node) = length(tree.left) + length(tree.right)
length(ensemble::Ensemble) = length(ensemble.trees)

depth(leaf::Leaf) = 0
depth(tree::Node) = 1 + max(depth(tree.left), depth(tree.right))

function print_tree(leaf::Leaf, depth=-1, indent=0)
    matches = find(leaf.values .== leaf.majority)
    ratio = string(length(matches)) * "/" * string(length(leaf.values))
    println("$(leaf.majority) : $(ratio)")
end

function print_tree(tree::Node, depth=-1, indent=0)
    if depth == indent
        println()
        return
    end
    println("Feature $(tree.featid), Threshold $(tree.featval)")
    print("    " ^ indent * "L-> ")
    print_tree(tree.left, depth, indent + 1)
    print("    " ^ indent * "R-> ")
    print_tree(tree.right, depth, indent + 1)
end

function show(io::IO, leaf::Leaf)
    println(io, "Decision Leaf")
    println(io, "Majority: $(leaf.majority)")
    print(io,   "Samples:  $(length(leaf.values))")
end

function show(io::IO, tree::Node)
    println(io, "Decision Tree")
    println(io, "Leaves: $(length(tree))")
    print(io,   "Depth:  $(depth(tree))")
end

function show(io::IO, ensemble::Ensemble)
    println(io, "Ensemble of Decision Trees")
    println(io, "Trees:      $(length(ensemble))")
    println(io, "Avg Leaves: $(mean([length(tree) for tree in ensemble.trees]))")
    print(io,   "Avg Depth:  $(mean([depth(tree) for tree in ensemble.trees]))")
end

end # module
