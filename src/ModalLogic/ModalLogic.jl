module ModalLogic

using IterTools
import Base: argmax, argmin, size, show, convert

using ComputedFieldTypes

export AbstractWorld, AbstractRelation,
				Ontology, OntologicalDataset,
				n_samples, n_variables, channel_size,
				MatricialInstance,
				MatricialDataset,
				MatricialUniDataset,
				WorldSet,
				display_propositional_test
				# , TestOperator
				# RelationAll, RelationNone, RelationId,
				# enumAcc,

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

################################################################################
# BEGIN Matricial dataset
################################################################################

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

n_samples(d::MatricialDataset{T,D})        where {T,D} = size(d, D-1)
n_variables(d::MatricialDataset{T,D})      where {T,D} = size(d, D)
channel_size(d::MatricialDataset{T,D})     where {T,D} = size(d)[1:end-2]

@computed struct OntologicalDataset{T,N}
	ontology  :: Ontology
	domain    :: MatricialDataset{T,N+1+1}
end

size(X::OntologicalDataset{T,N})             where {T,N} = size(X.domain)
size(X::OntologicalDataset{T,N}, i::Integer) where {T,N} = size(X.domain, i)
n_samples(X::OntologicalDataset{T,N})        where {T,N} = n_samples(X.domain)
n_variables(X::OntologicalDataset{T,N})      where {T,N} = n_variables(X.domain)
channel_size(X::OntologicalDataset{T,N})     where {T,N} = channel_size(X.domain)

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

# TODO use Xf[i,[(:) for i in 1:N]...]
# @computed @inline getFeature(X::OntologicalDataset{T,N}, idxs::AbstractVector{Integer}, feature::Integer) where T = X[idxs, feature, fill(:, N)...]::AbstractArray{T,N-1}

# TODO maybe using views can improve performances
# featureview(X::OntologicalDataset{T,0}, idxs::AbstractVector{Integer}, feature::Integer) = X.domain[idxs, feature]
# featureview(X::OntologicalDataset{T,1}, idxs::AbstractVector{Integer}, feature::Integer) = view(X.domain, idxs, feature, :)
# featureview(X::OntologicalDataset{T,2}, idxs::AbstractVector{Integer}, feature::Integer) = view(X.domain, idxs, feature, :, :)

################################################################################
# END Matricial dataset
################################################################################

################################################################################
# BEGIN Test operators
################################################################################

abstract type TestOperator end
# >=
struct _TestOpGeq  <: TestOperator end; const TestOpGeq  = _TestOpGeq();
# <
struct _TestOpLes  <: TestOperator end; const TestOpLes  = _TestOpLes();
# >=_α
struct _TestOpGeqSoft  <: TestOperator
  alpha :: AbstractFloat
end;
const TestOpGeq075  = _TestOpGeqSoft(0.75);
const TestOpGeq08  = _TestOpGeqSoft(0.8);
const TestOpGeq09  = _TestOpGeqSoft(0.9);
const TestOpGeq095  = _TestOpGeqSoft(0.95);
# <_α
struct _TestOpLesSoft  <: TestOperator
  alpha :: AbstractFloat
end;
const TestOpLes075  = _TestOpLesSoft(0.75);
const TestOpLes08  = _TestOpLesSoft(0.8);
const TestOpLes09  = _TestOpLesSoft(0.9);
const TestOpLes095  = _TestOpLesSoft(0.95);

display_propositional_test(test_operator::_TestOpGeq, lhs::String, featval::Number) = "$(lhs) >= $(featval)"
display_propositional_test(test_operator::_TestOpLes, lhs::String, featval::Number) = "$(lhs) < $(featval)"
display_propositional_test(test_operator::_TestOpGeqSoft, lhs::String, featval::Number) = "$(test_operator.alpha*100)% [$(lhs) >= $(featval)]"
display_propositional_test(test_operator::_TestOpLesSoft, lhs::String, featval::Number) = "$(test_operator.alpha*100)% [$(lhs) < $(featval)]"


@inline WMax(w::AbstractWorld, channel::MatricialChannel{T,N}) where {T,N} = maximum(readWorld(w,channel))
@inline WMin(w::AbstractWorld, channel::MatricialChannel{T,N}) where {T,N} = minimum(readWorld(w,channel))
@inline TestCondition(test_operator::_TestOpGeq, w::AbstractWorld, channel::MatricialChannel{T,N}, featval::Number) where {T,N} = begin # TODO maybe this becomes SIMD, or sum/all(readWorld(w,channel)  .<= featval)
	# Source: https://stackoverflow.com/questions/47564825/check-if-all-the-elements-of-a-julia-array-are-equal
	@inbounds for x in readWorld(w,channel)
		x >= featval || return false
	end
	return true
end
@inline TestCondition(test_operator::_TestOpLes, w::AbstractWorld, channel::MatricialChannel{T,N}, featval::Number) where {T,N} = begin # TODO maybe this becomes SIMD, or sum/all(readWorld(w,channel)  .<= featval)
	# Source: https://stackoverflow.com/questions/47564825/check-if-all-the-elements-of-a-julia-array-are-equal
	# @info "WLes" w featval #n readWorld(w,channel)
	@inbounds for x in readWorld(w,channel)
		x < featval || return false
	end
	return true
end

################################################################################
# END Test operators
################################################################################


# This constant is used to create the default world for each WorldType
#  (e.g. Interval(ModalLogic.emptyWorld) = Interval(-1,0))
struct _emptyWorld end;    const emptyWorld    = _emptyWorld();
struct _centeredWorld end; const centeredWorld = _centeredWorld();


# World generators/enumerators and array/set-like structures
# TODO test the functions for WorldSets with Sets and Arrays, and find the performance optimum
const AbstractWorldSet{W} = Union{AbstractVector{W},AbstractSet{W}} where {W<:AbstractWorld}
# Concrete type for sets: vectors are faster than sets, so we
# const WorldSet = AbstractSet{W} where W<:AbstractWorld
const WorldSet{W} = Vector{W} where {W<:AbstractWorld}
WorldSet{W}(S::WorldSet{W}) where {W<:AbstractWorld} = S

## Enumerate accessible worlds

# Fallback: enumAcc works with domains AND their dimensions
enumAcc(S::Any, r::AbstractRelation, X::MatricialChannel{T,N}) where {T,N} = enumAcc(S, r, size(X)...)
# Fallback: enumAcc for world sets maps to enumAcc-ing their elements
#  (note: one may overload this function to provide improved implementations for special cases (e.g. <L> of a world set in interval algebra))
enumAcc(S::AbstractWorldSet{worldType}, r::AbstractRelation, XYZ::Vararg{Integer,N}) where {T,N,worldType<:AbstractWorld} = begin
	IterTools.imap(worldType,
		IterTools.distinct(Iterators.flatten((enumAccBare(w, r, XYZ...) for w in S)))
	)
end

# Ontology-agnostic relations:
# - Identity relation  (RelationId)    =  S -> S
# - None relation      (RelationNone)  =  Used as the "nothing" constant
# - Universal relation (RelationAll)   =  S -> all-worlds
struct _RelationId    <: AbstractRelation end; const RelationId   = _RelationId();
struct _RelationNone  <: AbstractRelation end; const RelationNone = _RelationNone();
struct _RelationAll   <: AbstractRelation end; const RelationAll  = _RelationAll();

enumAcc(w::WorldType,           ::_RelationId, XYZ::Vararg{Integer,N}) where {WorldType<:AbstractWorld,N} = [w]
enumAcc(S::AbstractWorldSet{W}, ::_RelationId, XYZ::Vararg{Integer,N}) where {W<:AbstractWorld,N} = S # TODO try IterTools.imap(identity, S) ?
# Maybe this will have a use: enumAccW1(w::AbstractWorld, ::_RelationId,   X::Integer) where T = [w] # IterTools.imap(identity, [w])

print_rel_short(::_RelationId)  = "Id"
print_rel_short(::_RelationAll) = ""

# Perform the modal step, that is, evaluate a modal formula
#  on a domain, and eventually compute the new world set.
# TODO check that this dispatches on fastMode
modalStep(S::WorldSetType,
					relation::R,
					Xfi::AbstractArray{U,N},
					test_operator::Union{Val{true},Val{false}}, # TODO not boolean Union{Val{:<=},Val{:>}}
					threshold::U,
					fastMode::Val{V}) where {V, W<:AbstractWorld, WorldSetType<:Union{AbstractSet{W},AbstractVector{W}}, R<:AbstractRelation, U, N} = begin
	@info "modalStep"
	satisfied = false
	worlds = enumAcc(S, relation, Xfi)
	if length(collect(Iterators.take(worlds, 1))) > 0
		if fastMode == Val(false)
			new_worlds = WorldSetType()
		end
		for w in worlds # Sf[i]
			# @info " world" w
			# TODO make sure that the modal step doesn't require that all worlds contribute to the new set. Consider why <>(p and q) is different from (<>p and [](p->q)) 
			if (test_operator == Val(true) && TestCondition(TestOpGeq, w, Xfi, threshold)) ||
				 (test_operator == Val(false) && TestCondition(TestOpLes, w, Xfi, threshold))
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

end # module
