__precompile__()

module DecisionTree

import Base: length, show, convert, promote_rule, zero
using DelimitedFiles
using LinearAlgebra
import Random
using Statistics

export DTNode, DTLeaf, DTInternal,
			 is_leaf, is_modal_node,
			 num_nodes, height, modal_height,
			 build_stump, build_tree,
       print_tree, prune_tree, apply_tree,
			 ConfusionMatrix, confusion_matrix, mean_squared_error, R2, load_data,
			 #
			 startWithRelationAll, startAtCenter

# ScikitLearn API
export DecisionTreeClassifier,
#        DecisionTreeRegressor, RandomForestClassifier,
#        RandomForestRegressor, AdaBoostStumpClassifier,
#        # Should we export these functions? They have a conflict with
#        # DataFrames/RDataset over fit!, and users can always
#        # `using ScikitLearnBase`.
       predict,
       # predict_proba,
       fit!, get_classes

include("ModalLogic/ModalLogic.jl")
using .ModalLogic

###########################
########## Types ##########

abstract type _initCondition end
struct _startWithRelationAll  <: _initCondition end; const startWithRelationAll  = _startWithRelationAll();
struct _startAtCenter         <: _initCondition end; const startAtCenter         = _startAtCenter();

# Leaf node, holding the output decision
struct DTLeaf{T} # TODO specify output type: Number, Label, String, Union{Number,Label,String}?
	# Majority class/value (output)
	majority :: T
	# Training support
	values   :: Vector{T}
end

# Inner node, holding the output decision
struct DTInternal{S<:Real, T}

	modality :: R where R<:AbstractRelation
	featid   :: Int
	test_operator :: ModalLogic.TestOperator # Test operator (e.g. <=, ==)
	featval  :: featType where featType<:S
	# string representing an existential modality (e.g. ♢, L, LL)

	# Child nodes
	left     :: Union{DTLeaf{T}, DTInternal{S, T}}
	right    :: Union{DTLeaf{T}, DTInternal{S, T}}

end

# Decision node/tree # TODO figure out, maybe this has to be abstract and to supertype DTLeaf and DTInternal
const DTNode{S<:Real, T<:Real} = Union{DTLeaf{T}, DTInternal{S, T}}

struct DTree{S<:Real, T<:Real}
	root          :: DTNode{S, T}
	worldType     :: Type{<:AbstractWorld}
	initCondition :: _initCondition
end

is_leaf(l::DTLeaf) = true
is_leaf(n::DTInternal) = false
is_leaf(t::DTree) = is_leaf(t.root)

is_modal_node(n::DTInternal) = (!is_leaf(n) && n.modality != ModalLogic.RelationId)
is_modal_node(t::DTree) = is_modal_node(t.root)

zero(String) = ""
convert(::Type{DTInternal{S, T}}, lf::DTLeaf{T}) where {S, T} = DTInternal(ModalLogic.RelationNone, 0, :(nothing), zero(S), lf, DTLeaf(zero(T), [zero(T)]))

promote_rule(::Type{DTInternal{S, T}}, ::Type{DTLeaf{T}}) where {S, T} = DTInternal{S, T}

# make a Random Number Generator object
mk_rng(rng::Random.AbstractRNG) = rng
mk_rng(seed::T) where T <: Integer = Random.MersenneTwister(seed)

##############################
########## Includes ##########

include("measures.jl")
include("load_data.jl")
include("util.jl")
include("modal-classification/main.jl")
# TODO: include("ModalscikitlearnAPI.jl")

#############################
########## Methods ##########

# Length (total # of nodes)
num_nodes(leaf::DTLeaf) = 1
num_nodes(tree::DTInternal) = 1 + num_nodes(tree.left) + num_nodes(tree.right)
num_nodes(t::DTree) = num_nodes(t.root)

length(leaf::DTLeaf) = 1
length(tree::DTInternal) = length(tree.left) + length(tree.right)
length(t::DTree) = length(t.root)

# Height
height(leaf::DTLeaf) = 0
height(tree::DTInternal) = 1 + max(height(tree.left), height(tree.right))
height(t::DTree) = height(t.root)

# Modal height
modal_height(leaf::DTLeaf) = 0
modal_height(tree::DTInternal) = (is_modal_node(tree) ? 1 : 0) + max(modal_height(tree.left), modal_height(tree.right))
modal_height(t::DTree) = modal_height(t.root)

function print_tree(leaf::DTLeaf, depth=-1, indent=0, indent_guides=[])
		matches = findall(leaf.values .== leaf.majority)
		ratio = string(length(matches)) * "/" * string(length(leaf.values))
		println("$(leaf.majority) : $(ratio)")
end

function print_tree(tree::DTInternal, depth=-1, indent=0, indent_guides=[])
		if depth == indent
				println()
				return
		end

		println(ModalLogic.display_modal_test(tree.modality, tree.test_operator, tree.featid, tree.featval))
		# indent_str = " " ^ indent
		indent_str = reduce(*, [i == 1 ? "│" : " " for i in indent_guides])
		# print(indent_str * "╭✔")
		print(indent_str * "✔ ")
		print_tree(tree.left, depth, indent + 1, [indent_guides..., 1])
		# print(indent_str * "╰✘")
		print(indent_str * "✘ ")
		print_tree(tree.right, depth, indent + 1, [indent_guides..., 0])
end

function print_tree(tree::DTree)
		println("worldType: $(tree.worldType)")
		println("initCondition: $(tree.initCondition)")
		print_tree(tree.root)
end

function show(io::IO, leaf::DTLeaf)
		println(io, "Decision Leaf")
		println(io, "Majority: $(leaf.majority)")
		println(io, "Samples:  $(length(leaf.values))")
		print_tree(leaf)
end

function show(io::IO, tree::DTInternal)
		println(io, "Decision Node")
		println(io, "Leaves: $(length(tree))")
		println(io, "Tot nodes: $(num_nodes(tree))")
		println(io, "Height: $(height(tree))")
		println(io, "Modal height:  $(modal_height(tree))")
		print_tree(tree)
end

function show(io::IO, tree::DTree)
		println(io, "Decision Tree")
		println(io, "Leaves: $(length(tree))")
		println(io, "Tot nodes: $(num_nodes(tree))")
		println(io, "Height: $(height(tree))")
		println(io, "Modal height:  $(modal_height(tree))")
		print_tree(tree)
end

end # module


#=


# Function of the variables involved, returning a boolean 0/1 (= right/left)
# testFunc::TestFunction

abstract type ClassificationLeaf <: Leaf
    counts::Vector{Int}
    impurity::Float64
    n_samples::Int

    function ClassificationLeaf(example::Example, samples::Vector{Int}, impurity::Float64)
        counts = zeros(Int, example.n_labels)
        for s in samples
            label = example.y[s]
            counts[label] += round(Int, example.sample_weight[s])
        end
        new(counts, impurity, length(samples))
    end
end

# Decision function
abstract type TestFunction

struct PropositionalTestFunction <: TestFunction end
struct Modal∃TestFunction <: TestFunction end


other stuff...

struct OrderedPair
	x::Real
	y::Real
	OrderedPair(x,y) = x >= y ? error("out of order") : new(x,y)
end

# A wrapper around DataFrame to add labels to data.
type ClassificationDataset
	data::DataFrame
	labels::Array{Int,1}
end

# Inner node, holding the output decision
struct DTInternal <: DTNode
	
	# Child nodes
	children::Vector{DTNode}
	
	# Function of the variables involved, returning the index of the appropriate node given a value in the domain of the variable
	testFunc::TestFunction

end

=#
