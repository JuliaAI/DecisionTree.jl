__precompile__()

module DecisionTree

import Base: length, show, convert, promote_rule, zero
using DelimitedFiles
using LinearAlgebra
using Random
using Statistics

export Leaf, Node, Ensemble, print_tree, depth, build_stump, build_tree,
       prune_tree, apply_tree, apply_tree_proba, nfoldCV_tree, build_forest,
       apply_forest, apply_forest_proba, nfoldCV_forest, build_adaboost_stumps,
       apply_adaboost_stumps, apply_adaboost_stumps_proba, nfoldCV_stumps,
       majority_vote, ConfusionMatrix, confusion_matrix, mean_squared_error, R2, load_data

# ScikitLearn API
export DecisionTreeClassifier, DecisionTreeRegressor, RandomForestClassifier,
       RandomForestRegressor, AdaBoostStumpClassifier,
       # Should we export these functions? They have a conflict with
       # DataFrames/RDataset over fit!, and users can always
       # `using ScikitLearnBase`.
       predict, predict_proba, fit!, get_classes


###########################
########## Types ##########

struct Leaf{T}
    majority :: T
    values   :: Vector{T}
end

struct Node{S, T}
    featid  :: Int
    featval :: S
    left    :: Union{Leaf{T}, Node{S, T}}
    right   :: Union{Leaf{T}, Node{S, T}}
end

const LeafOrNode{S, T} = Union{Leaf{T}, Node{S, T}}

struct Ensemble{S, T}
    trees :: Vector{LeafOrNode{S, T}}
end

is_leaf(l::Leaf) = true
is_leaf(n::Node) = false

zero(String) = ""
convert(::Type{Node{S, T}}, lf::Leaf{T}) where {S, T} = Node(0, zero(S), lf, Leaf(zero(T), [zero(T)]))
promote_rule(::Type{Node{S, T}}, ::Type{Leaf{T}}) where {S, T} = Node{S, T}
promote_rule(::Type{Leaf{T}}, ::Type{Node{S, T}}) where {S, T} = Node{S, T}

# make a Random Number Generator object
mk_rng(rng::Random.AbstractRNG) = rng
mk_rng(seed::T) where T <: Integer = Random.MersenneTwister(seed)

##############################
########## Includes ##########

include("measures.jl")
include("load_data.jl")
include("util.jl")
include("classification/main.jl")
include("regression/main.jl")
include("scikitlearnAPI.jl")


#############################
########## Methods ##########

length(leaf::Leaf) = 1
length(tree::Node) = length(tree.left) + length(tree.right)
length(ensemble::Ensemble) = length(ensemble.trees)

depth(leaf::Leaf) = 0
depth(tree::Node) = 1 + max(depth(tree.left), depth(tree.right))

function print_tree(leaf::Leaf, depth=-1, indent=0; feature_names=nothing)
    matches = findall(leaf.values .== leaf.majority)
    ratio = string(length(matches)) * "/" * string(length(leaf.values))
    println("$(leaf.majority) : $(ratio)")
end

"""
       print_tree(tree::Node, depth=-1, indent=0; feature_names=nothing)

Print a textual visualization of the given decision tree `tree`.
In the example output below, the top node considers whether 
"Feature 3" is above or below the threshold -28.156052806422238.
If the value of "Feature 3" is below the threshold for some input to be classified, 
we move to the `L->` part underneath, which is a node 
looking at if "Feature 2" is above or below -161.04351901384842.
If the value of "Feature 2" is below the threshold for some input to be classified, 
we end up at `L-> 5 : 842/3650`. This is to be read as "In the left split, 
the tree will classify the input as class 5, as 842 of the 3650 datapoints 
in the training data that ended up here were of class 5."

# Example output:
```
Feature 3, Threshold -28.156052806422238
L-> Feature 2, Threshold -161.04351901384842
    L-> 5 : 842/3650
    R-> 7 : 2493/10555
R-> Feature 7, Threshold 108.1408338577021
    L-> 2 : 2434/15287
    R-> 8 : 1227/3508
```
"""
function print_tree(tree::Node, depth=-1, indent=0; feature_names=nothing)
    if depth == indent
        println()
        return
    end
    if feature_names === nothing
        println("Feature $(tree.featid), Threshold $(tree.featval)")
    else
        println("Feature $(tree.featid): \"$(feature_names[tree.featid])\", Threshold $(tree.featval)")
    end
    print("    " ^ indent * "L-> ")
    print_tree(tree.left, depth, indent + 1; feature_names = feature_names)
    print("    " ^ indent * "R-> ")
    print_tree(tree.right, depth, indent + 1; feature_names = feature_names)
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
