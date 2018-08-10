__precompile__()

module DecisionTree

import Base: length, show
using DelimitedFiles
using Distributed
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

function print_tree(leaf::Leaf, depth=-1, indent=0)
    matches = findall(leaf.values .== leaf.majority)
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
