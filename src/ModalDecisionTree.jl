__precompile__()

module ModalDecisionTree

import Base: length, show, convert, promote_rule, zero
using DelimitedFiles
using LinearAlgebra
using Random
using Statistics

export DTNode, DTLeaf, DTInternal,
			 is_leaf, is_modal_node,
			 height, modalHeight,
			 build_stump, build_tree,
       print_tree, prune_tree, apply_tree,
			 ConfusionMatrix, confusion_matrix, mean_squared_error, R2, load_data


###########################
########## Types ##########

# Leaf node, holding the output decision
struct DTLeaf{T} # TODO specify output type: Number, Label, String, Union{Number,Label,String}?
	# Majority class/value (output)
	majority :: T
	# Training support
	values   :: Vector{T}
end

# Inner node, holding the output decision
struct DTInternal{S<:Real, T}

	# Feature
	featid   :: Int
	featval  :: S
	testsign :: Symbol
	# string representing an existential modality (e.g. ♢, L, LL)
	modality :: Union{AbstractString,Nothing}

	# Child nodes
	left     :: Union{DTLeaf{T}, DTInternal{<:S, T}}
	right    :: Union{DTLeaf{T}, DTInternal{<:S, T}}

end

# Decision node/tree # TODO figure out, maybe this has to be abstract and to supertype DTLeaf and DTInternal
const DTNode{S<:Real, T<:Real} = Union{DTLeaf{T}, DTInternal{S, T}}

is_leaf(l::DTLeaf) = true
is_leaf(n::DTInternal) = false

is_modal_node(n::DTInternal) = n.modality isa Nothing

zero(String) = ""
convert(::Type{DTInternal{S, T}}, lf::DTLeaf{T}) where {S, T} = DTInternal(0, zero(S), lf, DTLeaf(zero(T), [zero(T)]))
promote_rule(::Type{DTInternal{S, T}}, ::Type{DTLeaf{T}}) where {S, T} = DTInternal{S, T}

# make a Random Number Generator object
mk_rng(rng::Random.AbstractRNG) = rng
mk_rng(seed::T) where T <: Integer = Random.MersenneTwister(seed)

##############################
########## Includes ##########

include("measures.jl")
include("load_data.jl")
include("util.jl")
include("ModalLogic.jl")
include("modal-classification/main.jl")
# TODO: include("ModalscikitlearnAPI.jl")


#############################
########## Methods ##########

# Length (total # of nodes)
length(leaf::DTLeaf) = 1
length(tree::DTInternal) = length(tree.left) + length(tree.right)

# Height
height(leaf::DTLeaf) = 0
height(tree::DTInternal) = 1 + max(height(tree.left), height(tree.right))

# Modal height
modalHeight(leaf::DTLeaf) = 0
modalHeight(tree::DTInternal) = (is_modal_node(tree) ? 1 : 0) + max(modalHeight(tree.left), modalHeight(tree.right))

function print_tree(leaf::DTLeaf, depth=-1, indent=0)
		matches = findall(leaf.values .== leaf.majority)
		ratio = string(length(matches)) * "/" * string(length(leaf.values))
		println("$(leaf.majority) : $(ratio)")
end

function print_tree(tree::DTInternal, depth=-1, indent=0)
		if depth == indent
				println()
				return
		end

		test = "Feature $(tree.featid) $(tree.testsign) $(tree.featval)"
		println(
			if ! ( is_modal_node(tree) )
				if tree.modality == "♢"
					modString = "$(tree.modality)"
				else
					modString = "<$(tree.modality)>"
				end
				"$modString ( $test )"
			else
				"$test"
			end)
		print("  " ^ indent * "Y-> ")
		print_tree(tree.left, depth, indent + 1)
		print("  " ^ indent * "N-> ")
		print_tree(tree.right, depth, indent + 1)
end

function show(io::IO, leaf::DTLeaf)
		println(io, "Decision Leaf")
		println(io, "Majority: $(leaf.majority)")
		print(io,   "Samples:  $(length(leaf.values))")
end

function show(io::IO, tree::DTInternal)
		println(io, "Decision Tree")
		println(io, "Leaves: $(length(tree))")
		print(io,   "Height:  $(height(tree))")
		print(io,   "Modal height:  $(modalHeight(tree))")
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
	labels::Array{Int}
end

# Inner node, holding the output decision
struct DTInternal <: DTNode
	
	# Child nodes
	children::Vector{DTNode}
	
	# Function of the variables involved, returning the index of the appropriate node given a value in the domain of the feature
	testFunc::TestFunction

end

=#
