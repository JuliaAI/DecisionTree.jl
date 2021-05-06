# __precompile__()

module DecisionTree

import Base: length, show, convert, promote_rule, zero
using DelimitedFiles
using LinearAlgebra
import Random
using Statistics

using Logging
using Logging: @logmsg
# Log single algorithm overview (e.g. splits performed in decision tree building)
const DTOverview = Logging.LogLevel(-500)
# Log debug info
const DTDebug = Logging.LogLevel(-1000)
# Log more detailed debug info
const DTDetail = Logging.LogLevel(-1500)

# TODO update these
export DTNode, DTLeaf, DTInternal,
			 is_leaf, is_modal_node,
			 num_nodes, height, modal_height,
			 build_stump, build_tree,
			 build_forest, apply_forest,
       print_tree, prune_tree, apply_tree, print_forest,
			 ConfusionMatrix, confusion_matrix, mean_squared_error, R2, load_data,
			 #
			 startWithRelationAll, startAtCenter,
			 DTOverview, DTDebug, DTDetail,
			 #
			 GammaType, GammaSliceType, spawn_rng


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

export print_apply_tree

include("ModalLogic/ModalLogic.jl")
using .ModalLogic
include("gammas.jl")
include("measures.jl")

###########################
########## Types ##########

abstract type _initCondition end
struct _startWithRelationAll  <: _initCondition end; const startWithRelationAll  = _startWithRelationAll();
struct _startAtCenter         <: _initCondition end; const startAtCenter         = _startAtCenter();
struct _startAtWorld{wT<:AbstractWorld} <: _initCondition w::wT end;

# Leaf node, holding the output decision
struct DTLeaf{T} # TODO specify output type: Number, Label, String, Union{Number,Label,String}?
	# Majority class/value (output)
	majority :: T
	# Training support
	values   :: Vector{T}
end

# Inner node, holding the output decision
struct DTInternal{S<:Real, T}
	# Split label
	modality      :: R where R<:AbstractRelation
	featid        :: Int
	test_operator :: TestOperator # Test operator (e.g. <=, ==)
	featval       :: featType where featType<:S
	# Child nodes
	left          :: Union{DTLeaf{T}, DTInternal{S, T}}
	right         :: Union{DTLeaf{T}, DTInternal{S, T}}

end

# Decision node/tree # TODO figure out, maybe this has to be abstract and to supertype DTLeaf and DTInternal
const DTNode{S<:Real, T} = Union{DTLeaf{T}, DTInternal{S, T}}

# TODO attach info about training (e.g. algorithm used + full namedtuple of training arguments) to these models
struct DTree{S<:Real, T}
	root          :: DTNode{S, T}
	worldType     :: Type{<:AbstractWorld}
	initCondition :: _initCondition
end

struct Forest{S<:Real, T}
	trees       :: Vector{Union{DTree{S, T},DTNode{S, T}}}
	cm          :: Vector{ConfusionMatrix}
	oob_error   :: AbstractFloat
end

is_leaf(l::DTLeaf) = true
is_leaf(n::DTInternal) = false
is_leaf(t::DTree) = is_leaf(t.root)

is_modal_node(n::DTInternal) = (!is_leaf(n) && n.modality != RelationId)
is_modal_node(t::DTree) = is_modal_node(t.root)

zero(String) = ""
convert(::Type{DTInternal{S, T}}, lf::DTLeaf{T}) where {S, T} = DTInternal(RelationNone, 0, :(nothing), zero(S), lf, DTLeaf(zero(T), [zero(T)]))

promote_rule(::Type{DTInternal{S, T}}, ::Type{DTLeaf{T}}) where {S, T} = DTInternal{S, T}

# make a Random Number Generator object
mk_rng(rng::Random.AbstractRNG) = rng
mk_rng(seed::T) where T <: Integer = Random.MersenneTwister(seed)

# Generate a new rng from a random pick from a given one.
spawn_rng(rng) = Random.MersenneTwister(abs(rand(rng, Int)))

##############################
########## Includes ##########

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
length(forest::Forest) = length(forest.trees)

# Height
height(leaf::DTLeaf) = 0
height(tree::DTInternal) = 1 + max(height(tree.left), height(tree.right))
height(t::DTree) = height(t.root)

# Modal height
modal_height(leaf::DTLeaf) = 0
modal_height(tree::DTInternal) = (is_modal_node(tree) ? 1 : 0) + max(modal_height(tree.left), modal_height(tree.right))
modal_height(t::DTree) = modal_height(t.root)

function print_tree(leaf::DTLeaf, depth=-1, indent=0, indent_guides=[]; n_tot_inst = false)
		matches = findall(leaf.values .== leaf.majority)

		n_correct =length(matches)
		n_inst = length(leaf.values)

		confidence = n_correct/n_inst
		
		metrics = "conf: $(confidence)"
		
		if n_tot_inst != false
			support = n_inst/n_tot_inst
			metrics *= ", supp = $(support)"
			# lift = ...
			# metrics *= ", lift = $(lift)"
			# conv = ...
			# metrics *= ", conv = $(conv)"
		end

		println("$(leaf.majority) : $(n_correct)/$(n_inst) ($(metrics))") # TODO print purity?
end

function print_tree(tree::DTInternal, depth=-1, indent=0, indent_guides=[]; n_tot_inst = false)
		if depth == indent
				println()
				return
		end

		println(display_modal_test(tree.modality, tree.test_operator, tree.featid, tree.featval)) # TODO print purity?
		# indent_str = " " ^ indent
		indent_str = reduce(*, [i == 1 ? "│" : " " for i in indent_guides])
		# print(indent_str * "╭✔")
		print(indent_str * "✔ ")
		print_tree(tree.left, depth, indent + 1, [indent_guides..., 1], n_tot_inst = n_tot_inst)
		# print(indent_str * "╰✘")
		print(indent_str * "✘ ")
		print_tree(tree.right, depth, indent + 1, [indent_guides..., 0], n_tot_inst = n_tot_inst)
end

function print_tree(tree::DTree; n_tot_inst = false)
		println("worldType: $(tree.worldType)")
		println("initCondition: $(tree.initCondition)")
		print_tree(tree.root, n_tot_inst = n_tot_inst)
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


function print_forest(forest::Forest)
	n_trees = length(forest)
	for i in 1:n_trees
		println("Tree $(i) / $(n_trees)")
		print_tree(forest.trees[i])
	end
end

function show(io::IO, forest::Forest)
		println(io, "Forest")
		println(io, "Num trees: $(length(forest))")
		println(io, "Out-Of-Bag Error: $(forest.oob_error)")
		println(io, "ConfusionMatrix: $(forest.cm)")
		println(io, "Trees:")
		print_forest(forest)
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

# Inner node, holding the output decision
struct DTInternal <: DTNode
	
	# Child nodes
	children::Vector{DTNode}
	
	# Function of the variables involved, returning the index of the appropriate node given a value in the domain of the variable
	testFunc::TestFunction

end

=#
