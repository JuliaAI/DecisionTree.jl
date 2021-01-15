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
				IntervalOntology,
				getfeature
				# RelationAll, RelationNone, RelationId,
				# enumAcc,
				# IARelations,

# Fix
Base.keys(g::Base.Generator) = g.iter

# Abstract classes for world & relations
abstract type AbstractWorld end
abstract type AbstractRelation end

# Concrete class for ontology models (world type + set of relations)
struct Ontology
	worldType   :: Type{<:AbstractWorld}
	relationSet :: AbstractVector{<:AbstractRelation}
end

# This constant is used to create the default world for each WorldType
#  (e.g. Interval(ModalLogic.InitialWorld) = Interval(-1,0))
struct _InitialWorld end; const InitialWorld = _InitialWorld();

# A dataset, given by a set of N-dimensional (multi-variate) matrices/instances,
#  and an Ontology to be interpreted on each of them.
# - The size of the domain is n_samples × n_variables × [Matricial domain]
# - Note that N is the dimension of the dimensional domain itself
#    (e.g. 1 for the temporal case, 2 for the spatialCase)
#  https://discourse.julialang.org/t/addition-to-parameter-of-parametric-type/20059/5
@computed struct OntologicalDataset{T,N}
	ontology  :: Ontology
	domain    :: AbstractArray{T,N+2}
end

size(X::OntologicalDataset{T,N}) where {T,N} = size(X.domain)
size(X::OntologicalDataset{T,N}, n::Integer) where {T,N} = size(X.domain, n)
n_samples(X::OntologicalDataset{T,N}) where {T,N} = size(X, 1)
n_variables(X::OntologicalDataset{T,N}) where {T,N} = size(X, 2)
dimension(X::OntologicalDataset{T,N}) where {T,N} = size(X, 1)-2

@inline getfeature(X::OntologicalDataset{T,0}, idx::Integer, feature::Integer) where T = X.domain[idx, feature]::T
@inline getfeature(X::OntologicalDataset{T,1}, idx::Integer, feature::Integer) where T = X.domain[idx, feature, :]::AbstractArray{T,1}
@inline getfeature(X::OntologicalDataset{T,2}, idx::Integer, feature::Integer) where T = X.domain[idx, feature, :, :]::AbstractArray{T,2}
# @computed @inline getfeature(X::OntologicalDataset{T,N}, idxs::AbstractVector{Integer}, feature::Integer) where T = X.domain[idxs, feature, fill(:, N)...]::AbstractArray{T,N-1}
# @computed @inline getfeature(X::OntologicalDataset{T,N}, idxs::AbstractVector{Integer}, feature::Integer) where T = X.domain[idxs, feature, fill(:, dimension(X))...]::AbstractArray{T,N-1}

# TODO maybe using views can improve performances
# featureview(X::OntologicalDataset{T,0}, idxs::AbstractVector{Integer}, feature::Integer) = X.domain[idxs, feature]
# featureview(X::OntologicalDataset{T,1}, idxs::AbstractVector{Integer}, feature::Integer) = view(X.domain, idxs, feature, :)
# featureview(X::OntologicalDataset{T,2}, idxs::AbstractVector{Integer}, feature::Integer) = view(X.domain, idxs, feature, :, :)

@inline getslice(Xf::AbstractArray{T, 1}, idx::Integer) where T = Xf[idx] # TODO: the adimensional case return a value of type T, instead of an array. Mh, fix? Or we could say we don't handl the adimensional case
@inline getslice(Xf::AbstractArray{T, 2}, idx::Integer) where T = Xf[idx,:]
@inline getslice(Xf::AbstractArray{T, 3}, idx::Integer) where T = Xf[idx,:,:]
# TODO use Xf[i,[(:) for i in 1:N]...]

# World generators/enumerators and array/set-like structures
# TODO test the functions for WorldSets with Sets and Arrays, and find the performance optimum
const WorldGenerator = Union{Base.Generator,IterTools.Distinct}
const WorldSet{T} = Union{AbstractArray{T,1},AbstractSet{T}} where {T<:AbstractWorld}

# Ontology-agnostic relations:
# - Identity relation  (RelationId)    =  S -> S
# - None relation      (RelationNone)  =  Used as the "nothing" constant
# - Universal relation (RelationAll)   =  S -> all-worlds
struct _RelationId    <: AbstractRelation end; const RelationId   = _RelationId();
struct _RelationNone  <: AbstractRelation end; const RelationNone = _RelationNone();
struct _RelationAll   <: AbstractRelation end; const RelationAll  = _RelationAll();

enumAcc(S::Union{WorldGenerator,WorldSet{<:AbstractWorld}}, ::_RelationId, X::AbstractArray{T,1}) where T = S # IterTools.imap(identity, S)
# Maybe this will have a use: enumAccW1(w::AbstractWorld, ::_RelationId,   X::AbstractArray{T,1}) where T = [w] # IterTools.imap(identity, [w])

# Perform the modal step, that is, evaluate a modal formula
#  on a domain, and eventually compute the new world set.
# TODO check that this dispatches on fastMode
modalStep(S::AbstractSet{W},
					Xfi::AbstractArray{U,N},
					relation::R,
					threshold::U,
					fastMode::Val{V} where V) where {W<:AbstractWorld, R<:AbstractRelation, U, N} = begin
	@info "modalStep"
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
