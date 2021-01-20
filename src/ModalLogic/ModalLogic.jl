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
				MatricialDataset,
				MatricialUniDataset,
				WorldSet
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
	Ontology(worldType, relationSet) = new(worldType, collect(Set(relationSet)))
end

# This constant is used to create the default world for each WorldType
#  (e.g. Interval(ModalLogic.InitialWorld) = Interval(-1,0))
struct _InitialWorld end; const InitialWorld = _InitialWorld();

# A dataset, given by a set of N-dimensional (multi-variate) matrices/instances,
#  and an Ontology to be interpreted on each of them.
# - The size of the domain array is {X×Y×...} × n_samples × n_variables
# - N is the dimensionality of the domain itself (e.g. 1 for the temporal case, 2 for the spatialCase)
#    and its dimensions are denoted as X,Y,Z,...
# - A uni-variate dataset is of dimensionality S=N+1
# - A multi-variate dataset is of dimensionality D=N+1+1
#  https://discourse.julialang.org/t/addition-to-parameter-of-parametric-type/20059/5

const MatricialChannel{T,N}   = AbstractArray{T,N}
const MatricialInstance{T,MN} = AbstractArray{T,MN}
# TODO: It'd be nice to define these as a function of N, https://github.com/JuliaLang/julia/issues/8322
#   e.g. const MatricialUniDataset{T,N}       = AbstractArray{T,N+1}
const MatricialUniDataset{T,UD} = AbstractArray{T,UD}
const MatricialDataset{T,D}     = AbstractArray{T,D}

@computed struct OntologicalDataset{T,N}
	ontology  :: Ontology
	domain    :: MatricialDataset{T,N+1+1}
end

size(X::OntologicalDataset{T,N})             where {T,N} = size(X.domain)
size(X::OntologicalDataset{T,N}, i::Integer) where {T,N} = size(X.domain, i)
n_samples(X::OntologicalDataset{T,N})        where {T,N} = size(X, N+1)
n_variables(X::OntologicalDataset{T,N})      where {T,N} = size(X, N+2)
# dimensionality(X::OntologicalDataset{T,N})   where {T,N} = N

# TODO: the adimensional case returns a value of type T, instead of an array. Mh, fix? Or we could say we don't handl the adimensional case
@inline getChannel(ud::MatricialUniDataset{T,1},  idx::Integer) where T = ud[idx]           # N=0
@inline getChannel(ud::MatricialUniDataset{T,2},  idx::Integer) where T = ud[:, idx]        # N=1
@inline getChannel(ud::MatricialUniDataset{T,3},  idx::Integer) where T = ud[:, :, idx]     # N=2
@inline getInstance(d::MatricialDataset{T,2},     idx::Integer) where T = d[idx, :]         # N=0
@inline getInstance(d::MatricialDataset{T,3},     idx::Integer) where T = d[:, idx, :]      # N=1
@inline getInstance(d::MatricialDataset{T,4},     idx::Integer) where T = d[:, :, idx, :]   # N=2
@inline getInstanceFeature(instance::MatricialInstance{T,1},      idx::Integer) where T = instance[      idx]::T                      # N=0
@inline getInstanceFeature(instance::MatricialInstance{T,2},      idx::Integer) where T = instance[:,    idx]::MatricialChannel{T,1}  # N=1
@inline getInstanceFeature(instance::MatricialInstance{T,3},      idx::Integer) where T = instance[:, :, idx]::MatricialChannel{T,2}  # N=2
@inline getFeature(d::MatricialDataset{T,2},      idx::Integer, feature::Integer) where T = d[      idx, feature]::T                     # N=0
@inline getFeature(d::MatricialDataset{T,3},      idx::Integer, feature::Integer) where T = d[:,    idx, feature]::MatricialChannel{T,1} # N=1
@inline getFeature(d::MatricialDataset{T,4},      idx::Integer, feature::Integer) where T = d[:, :, idx, feature]::MatricialChannel{T,2} # N=2

# TODO generalize as init_Xf(X::OntologicalDataset{T, N}) where T = Array{T, N+1}(undef, size(X)[3:end]..., n_samples(X))
# Initialize MatricialUniDataset by slicing across the features dimension
MatricialUniDataset(::UndefInitializer, d::MatricialDataset{T,2}) where T = Array{T, 1}(undef, n_samples(d))::MatricialUniDataset{T, 1}
MatricialUniDataset(::UndefInitializer, d::MatricialDataset{T,3}) where T = Array{T, 2}(undef, size(d)[1:end-1])::MatricialUniDataset{T, 2}
MatricialUniDataset(::UndefInitializer, d::MatricialDataset{T,4}) where T = Array{T, 3}(undef, size(d)[1:end-1])::MatricialUniDataset{T, 3}

# @computed @inline getFeature(X::OntologicalDataset{T,N}, idxs::AbstractVector{Integer}, feature::Integer) where T = X[idxs, feature, fill(:, N)...]::AbstractArray{T,N-1}
# @computed @inline getFeature(X::OntologicalDataset{T,N}, idxs::AbstractVector{Integer}, feature::Integer) where T = X[idxs, feature, fill(:, dimensionality(X))...]::AbstractArray{T,N-1}

# TODO maybe using views can improve performances
# featureview(X::OntologicalDataset{T,0}, idxs::AbstractVector{Integer}, feature::Integer) = X.domain[idxs, feature]
# featureview(X::OntologicalDataset{T,1}, idxs::AbstractVector{Integer}, feature::Integer) = view(X.domain, idxs, feature, :)
# featureview(X::OntologicalDataset{T,2}, idxs::AbstractVector{Integer}, feature::Integer) = view(X.domain, idxs, feature, :, :)

# TODO use Xf[i,[(:) for i in 1:N]...]

# World generators/enumerators and array/set-like structures
# TODO test the functions for WorldSets with Sets and Arrays, and find the performance optimum
const WorldGenerator = Union{Base.Generator,IterTools.Distinct}
const AbstractWorldSet{W} = Union{AbstractVector{W},AbstractSet{W}} where {W<:AbstractWorld}
# Concrete type for sets: vectors are faster than sets, so we
# const WorldSet = AbstractSet{W} where W<:AbstractWorld
const WorldSet{W} = Vector{W} where {W<:AbstractWorld}
WorldSet{W}(S::WorldSet{W}) where {W<:AbstractWorld} = S

# Ontology-agnostic relations:
# - Identity relation  (RelationId)    =  S -> S
# - None relation      (RelationNone)  =  Used as the "nothing" constant
# - Universal relation (RelationAll)   =  S -> all-worlds
struct _RelationId    <: AbstractRelation end; const RelationId   = _RelationId();
struct _RelationNone  <: AbstractRelation end; const RelationNone = _RelationNone();
struct _RelationAll   <: AbstractRelation end; const RelationAll  = _RelationAll();

enumAcc(S::Union{WorldGenerator,AbstractWorldSet{<:AbstractWorld}}, ::_RelationId, X::AbstractArray{T,1}) where T = S # IterTools.imap(identity, S)
# Maybe this will have a use: enumAccW1(w::AbstractWorld, ::_RelationId,   X::AbstractArray{T,1}) where T = [w] # IterTools.imap(identity, [w])

print_rel_short(::_RelationId)  = "Id"
print_rel_short(::_RelationAll) = ""

# Perform the modal step, that is, evaluate a modal formula
#  on a domain, and eventually compute the new world set.
# TODO check that this dispatches on fastMode
modalStep(S::WorldSetType,
					Xfi::AbstractArray{U,N},
					relation::R,
					threshold::U,
					fastMode::Val{V} where V) where {W<:AbstractWorld, WorldSetType<:Union{AbstractSet{W},AbstractVector{W}}, R<:AbstractRelation, U, N} = begin
	@info "modalStep"
	satisfied = false
	worlds = enumAcc(S, relation, Xfi)
	if length(collect(Iterators.take(worlds, 1))) > 0
		# TODO maybe it's better to use an array and then create a set with = Set(worlds)
		if fastMode == Val(false)
			new_worlds = WorldSetType()
		end
		for w in worlds # Sf[i]
			# @info " world" w
			if WLeq(w, Xfi, threshold) # WLeq is <= TODO expand on test_operator
				satisfied = true
				if fastMode == Val(false)
					push!(new_worlds, w)
				elseif fastMode == Val(true)
					@info "   Found w: " w # readWorld(w,Xfi)
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

end # module
