module ModalLogic

using IterTools
import Base.argmax
import Base.argmin
import Base.size
import Base.show

using ComputedFieldTypes

export AbstractWorld, AbstractRelation,
				Ontology, OntologicalDataset,
				n_samples, n_variables, dimension,
				getfeature,
				WorldGenerator,
				# RelationAll, RelationNone, RelationEq,
				# enumAcc,
				# readMax,
				# readMin,
				# IARelations,
				IntervalOntology

# Fix
Base.keys(g::Base.Generator) = g.iter

# # Generic Kripke frame: worlds & relations
# struct KripkeFrame{T} <: AbstractKripkeFrame{T}
# 	# Majority class/value (output)
# 	worlds :: AbstractVector{T}
# 	# Training support
# 	values :: Vector{T}
# end

# Abstract classes for world & relations
abstract type AbstractWorld end
abstract type AbstractRelation end
# abstract type AbstractKripkeFrame end

# Concrete class for ontology models
struct Ontology
	worldType   :: Type{<:AbstractWorld}
	relationSet :: AbstractArray{<:AbstractRelation}
end

struct _InitialWorld end; const InitialWorld = _InitialWorld();


# An ontology interpreted over an N-dimensional domain gives rise to a Kripke model/frame.
# const MatricialDomain{T,N} = AbstractArray{T,N} end
# struct OntologicalKripkeFrame{T,N}
# 	ontology  :: Ontology
# 	domain    :: AbstractArray{T,N}
# end

# A dataset, given by a set of N-dimensional (multi-variate) matrices/instances,
#  and an Ontology to be interpreted on each of them
# Note that N is the dimension of the dimensional domain itself (e.g. 0 for the adimensional case, 1 for the temporal case)
#  https://discourse.julialang.org/t/addition-to-parameter-of-parametric-type/20059/5
@computed struct OntologicalDataset{T,N}
	ontology  :: Ontology
	domain    :: AbstractArray{T,N+2}
end

# TODO use staticArrays for small images https://github.com/JuliaArrays/StaticArrays.jl
#  X = OntologicalDataset(IntervalOntology,Array{SMatrix{3,3,Int},2}(undef, 20, 10))

# TODO maybe the domain should not be 20x3x3 but 3x3x20, because Julia is column-wise
size(X::OntologicalDataset{T,N}) where {T,N} = size(X.domain)
size(X::OntologicalDataset{T,N}, n::Integer) where {T,N} = size(X.domain, n)
n_samples(X::OntologicalDataset{T,N}) where {T,N} = size(X, 1)
n_variables(X::OntologicalDataset{T,N}) where {T,N} = size(X, 2)
dimension(X::OntologicalDataset{T,N}) where {T,N} = size(X, 1)-2

@inline getslice(Xf::AbstractArray{T, 1}, idx::Integer) where T = Xf[idx] # TODO: the adimensional case return a value of type T, instead of an array. Mh, fix? Or we could say we don't handl the adimensional case
@inline getslice(Xf::AbstractArray{T, 2}, idx::Integer) where T = Xf[idx,:]
@inline getslice(Xf::AbstractArray{T, 3}, idx::Integer) where T = Xf[idx,:,:]
# TODO use Xf[i,[: for i in N]...]

@inline getfeature(X::OntologicalDataset{T,0}, idx::Integer, feature::Integer) where T = X.domain[idx, feature]::T
@inline getfeature(X::OntologicalDataset{T,1}, idx::Integer, feature::Integer) where T = X.domain[idx, feature, :]::AbstractArray{T,1}
@inline getfeature(X::OntologicalDataset{T,2}, idx::Integer, feature::Integer) where T = X.domain[idx, feature, :, :]::AbstractArray{T,2}
# @computed @inline getfeature(X::OntologicalDataset{T,N}, idxs::AbstractVector{Integer}, feature::Integer) where T = X.domain[idxs, feature, fill(:, N)...]::AbstractArray{T,N-1}
# @computed @inline getfeature(X::OntologicalDataset{T,N}, idxs::AbstractVector{Integer}, feature::Integer) where T = X.domain[idxs, feature, fill(:, dimension(X))...]::AbstractArray{T,N-1}

const WorldGenerator = Union{Base.Generator,IterTools.Distinct}
# TODO test the functions for WorldSets with Sets and Arrays, and find the performance optimum
const WorldSet{T} = Union{AbstractArray{T,1},AbstractSet{T}}  


# Equality relation, that exists for every Ontology

struct _RelationEq    <: AbstractRelation end; const RelationEq   = _RelationEq();
struct _RelationNone  <: AbstractRelation end; const RelationNone = _RelationNone();
struct _RelationAll   <: AbstractRelation end; const RelationAll  = _RelationAll();

enumAcc(S::Union{WorldGenerator,WorldSet{<:AbstractWorld}}, ::_RelationEq, X::AbstractArray{T,1}) where T = S # IterTools.imap(identity, S)
# Maybe this will have a use: enumAccW1(w::AbstractWorld, ::_RelationEq,   X::AbstractArray{T,1}) where T = [w] # IterTools.imap(identity, [w])

# TODO maybe using views can improve performances
# featureview(X::OntologicalDataset{T,0}, idxs::AbstractVector{Integer}, feature::Integer) = X.domain[idxs, feature]
# featureview(X::OntologicalDataset{T,1}, idxs::AbstractVector{Integer}, feature::Integer) = view(X.domain, idxs, feature, :)
# featureview(X::OntologicalDataset{T,2}, idxs::AbstractVector{Integer}, feature::Integer) = view(X.domain, idxs, feature, :, :)

# In the most generic case, a Kripke model/frame can be reprented in graph form.
# Thus, an "AbstractKripkeFrame" should also supertype some other representation.

# TODO Generalize World as a tuple of parameters ( https://stackoverflow.com/questions/40160120/generic-constructors-for-subtypes-of-an-abstract-type )

# TODO check that this dispatches on fastMode
modalStep(S::AbstractSet{W}, Xfi::AbstractArray{U,N}, relation::R where R<:AbstractRelation, threshold::U, fastMode::Val{V} where V) where {U, N, W<:AbstractWorld} = begin
	@info " modalStep"
	satisfied = false
	worlds = enumAcc(S, relation, Xfi)
	if length(collect(Iterators.take(worlds, 1))) > 0
		# TODO maybe it's better to use an array and then create a set with = Set(worlds)
		if fastMode == Val(false)
			new_worlds = Set{W}()
		end
		for w in worlds # Sf[i]
			# @info " world" w
			if WLeq(w, Xfi, threshold) # WLeq is <= TODO expand on testsign
				satisfied = true
				if fastMode == Val(false)
					push!(new_worlds, w)
				elseif fastMode == Val(true)
					@info "   Found w: " w readWorld(w,Xfi)
					break
				end
			end
		end
		if fastMode == Val(false)
			if satisfied == true
				S = new_worlds
			else 
				# If none of the neighboring worlds satisfies the propositional condition, then 
				#  the new set is left unchanged
			end
		end
	else
		@info "   No world found"
		# If there are no neighboring worlds, then the modal condition is not met
	end
	if satisfied
		@info "   YES"
	else
		@info "   NO" 
	end
	return (satisfied,S)
end

include("IntervalLogic.jl")

getParallelOntologyOfDim(::Val{1}) = IntervalOntology
# getParallelOntologyOfDim(::Val{2}) = RectangleAlgebra
#=
############################################
END Performance tuning
############################################


using Revise
using BenchmarkTools
include("DecisionTree.jl/src/ModalLogic.jl")


X = fill(1, 40)
S = [Interval(15, 25)]
S1 = enumAcc1(S, IA_L, X)
S2 = enumAcc2(S, IA_L, X)
Sc = Array{Interval,1}(collect(S))


@btime enumAcc(S1, IA_L,  X) |> collect;
@btime enumAcc(S2, IA_L,  X) |> collect;
@btime enumAcc(Sc, IA_L,  X) |> collect;
@btime enumAcc(S1, IA_Di,  X) |> collect;
@btime enumAcc(S2, IA_Di,  X) |> collect;
@btime enumAcc(Sc, IA_Di,  X) |> collect;
@btime enumAcc(S1, IA_Oi,  X) |> collect;
@btime enumAcc(S2, IA_Oi,  X) |> collect;
@btime enumAcc(Sc, IA_Oi,  X) |> collect;

=#

#=
TODO next
# 2D Interval counterpart Rectangle parallel
# struct ParRectangle <: AbstractWorld
# 	h :: Interval
# 	v :: Interval
# end

# const RectangleAlgebra = Ontology(ParRectangle,RARelation)

=#

end # module
