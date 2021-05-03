module ModalLogic

using IterTools
import Base: argmax, argmin, size, show, convert
using Logging: @logmsg
using ..DecisionTree

using ComputedFieldTypes

export AbstractWorld, AbstractRelation,
				Ontology, OntologicalDataset,
				n_samples, n_variables, channel_size,
				MatricialInstance,
				MatricialDataset,
				MatricialUniDataset,
				WorldSet,
				display_propositional_test,
				display_modal_test
				# , TestOperator
				# RelationAll, RelationNone, RelationId,
				# enumAccessibles, enumAccRepr

# Fix
Base.keys(g::Base.Generator) = g.iter

# Abstract classes for world & relations
abstract type AbstractWorld end
abstract type AbstractRelation end

# This constant is used to create the default world for each WorldType
#  (e.g. Interval(ModalLogic.emptyWorld) = Interval(-1,0))
struct _firstWorld end;    const firstWorld    = _firstWorld();
struct _emptyWorld end;    const emptyWorld    = _emptyWorld();
struct _centeredWorld end; const centeredWorld = _centeredWorld();

# One unique world
struct OneWorld    <: AbstractWorld
	OneWorld() = new()
	OneWorld(w::_emptyWorld) = new()
	OneWorld(w::_firstWorld) = new()
	OneWorld(w::_centeredWorld) = new()
end;

show(io::IO, w::OneWorld) = begin
	print(io, "−")
end

worldTypeDimensionality(::Type{OneWorld}) = 0
print_world(::OneWorld) = println("−")

# Concrete class for ontology models (world type + set of relations)
struct Ontology
	worldType   :: Type{<:AbstractWorld}
	relationSet :: AbstractVector{<:AbstractRelation}
	Ontology(worldType, relationSet) = begin
		relationSet = unique(relationSet)
		for relation in relationSet
			if !goesWith(worldType, relation)
				error("Can't instantiate Ontology with worldType $(worldType) and relation $(relation)")
			end
		end
		return new(worldType, relationSet)
	end
	# Ontology(worldType, relationSet) = new(worldType, relationSet)
end

strip_ontology(ontology::Ontology) = Ontology(OneWorld,AbstractRelation[])

# TODO improve, decouple from relationSets definitions
# Actually, this will not work because relationSet does this collect(set(...)) thing... mh maybe better avoid that thing?
show(io::IO, o::Ontology) = begin
	print(io, "Ontology(")
	show(io, o.worldType)
	print(io, ",")
	if issetequal(o.relationSet, IARelations)
		print(io, "IARelations")
	elseif issetequal(o.relationSet, IARelations_extended)
		print(io, "IARelations_extended")
	elseif issetequal(o.relationSet, IA2DRelations)
		print(io, "IA2DRelations")
	elseif issetequal(o.relationSet, IA2DRelations_U)
		print(io, "IA2DRelations_U")
	elseif issetequal(o.relationSet, IA2DRelations_extended)
		print(io, "IA2DRelations_extended")
	elseif issetequal(o.relationSet, RCC8Relations)
		print(io, "RCC8")
	elseif issetequal(o.relationSet, RCC5Relations)
		print(io, "RCC5")
	else
		show(io, o.relationSet)
	end
	print(io, ")")
end

################################################################################
# BEGIN Helpers
################################################################################

# https://stackoverflow.com/questions/46671965/printing-variable-subscripts-in-julia/46674866
# '₀'
subscriptnumber(i::Int) = begin
	join([
		(if i < 0
			[Char(0x208B)]
		else [] end)...,
		[Char(0x2080+d) for d in reverse(digits(abs(i)))]...
	])
end
# https://www.w3.org/TR/xml-entity-names/020.html
# '․', 'ₑ', '₋'
subscriptnumber(s::AbstractString) = begin
	char_to_subscript(ch) = begin
		if ch == 'e'
			'ₑ'
		elseif ch == '.'
			'․'
		elseif ch == '.'
			'․'
		elseif ch == '-'
			'₋'
		else
			subscriptnumber(parse(Int, ch))
		end
	end

	try
		join(map(char_to_subscript, [string(ch) for ch in s]))
	catch
		s
	end
end

subscriptnumber(i::AbstractFloat) = subscriptnumber(string(i))

################################################################################
# END Helpers
################################################################################

################################################################################
# BEGIN Matricial dataset & Ontological dataset
################################################################################

# A dataset, given by a set of N-dimensional (multi-variate) matrices/instances,
#  and an Ontology to be interpreted on each of them.
# - The size of the domain array is {X×Y×...} × n_samples × n_variables
# - N is the dimensionality of the domain itself (e.g. 1 for the temporal case, 2 for the spatialCase)
#    and its dimensions are denoted as X,Y,Z,...
# - A uni-variate dataset is of dimensionality S=N+1
# - A multi-variate dataset is of dimensionality D=N+1+1
#  https://discourse.julialang.org/t/addition-to-parameter-of-parametric-type/20059/5

const MatricialChannel{T,N}     = AbstractArray{T,N}
const MatricialInstance{T,MN}   = AbstractArray{T,MN}
# TODO: It'd be nice to define these as a function of N, https://github.com/JuliaLang/julia/issues/8322
#   e.g. const MatricialUniDataset{T,N}       = AbstractArray{T,N+1}
const MatricialUniDataset{T,UD} = AbstractArray{T,UD}
const MatricialDataset{T,D}     = AbstractArray{T,D}

n_samples(d::MatricialDataset{T,D})    where {T,D} = size(d, D-1)
n_variables(d::MatricialDataset{T,D})  where {T,D} = size(d, D)
channel_size(d::MatricialDataset{T,D}) where {T,D} = size(d)[1:end-2]

@inline getInstance(d::MatricialDataset{T,2},     idx::Integer) where T = @views d[idx, :]         # N=0
@inline getInstance(d::MatricialDataset{T,3},     idx::Integer) where T = @views d[:, idx, :]      # N=1
@inline getInstance(d::MatricialDataset{T,4},     idx::Integer) where T = @views d[:, :, idx, :]   # N=2
@inline getFeature(d::MatricialDataset{T,2},      idx_i::Integer, idx_f::Integer) where T = @views d[      idx_i, idx_f]::T                     # N=0
@inline getFeature(d::MatricialDataset{T,3},      idx_i::Integer, idx_f::Integer) where T = @views d[:,    idx_i, idx_f]::MatricialChannel{T,1} # N=1
@inline getFeature(d::MatricialDataset{T,4},      idx_i::Integer, idx_f::Integer) where T = @views d[:, :, idx_i, idx_f]::MatricialChannel{T,2} # N=2
@inline getChannel(ud::MatricialUniDataset{T,1},  idx::Integer) where T = @views ud[idx]           # N=0
@inline getChannel(ud::MatricialUniDataset{T,2},  idx::Integer) where T = @views ud[:, idx]        # N=1
@inline getChannel(ud::MatricialUniDataset{T,3},  idx::Integer) where T = @views ud[:, :, idx]     # N=2
@inline getInstanceFeature(inst::MatricialInstance{T,1},      idx::Integer) where T = @views inst[      idx]::T                     # N=0
@inline getInstanceFeature(inst::MatricialInstance{T,2},      idx::Integer) where T = @views inst[:,    idx]::MatricialChannel{T,1} # N=1
@inline getInstanceFeature(inst::MatricialInstance{T,3},      idx::Integer) where T = @views inst[:, :, idx]::MatricialChannel{T,2} # N=2

@inline sliceDomainByInstances(d::MatricialDataset{T,2}, inds::AbstractVector{<:Integer}; return_view = false) where T = if return_view @views d[inds, :]       else d[inds, :]    end # N=0
@inline sliceDomainByInstances(d::MatricialDataset{T,3}, inds::AbstractVector{<:Integer}; return_view = false) where T = if return_view @views d[:, inds, :]    else d[:, inds, :] end # N=1
@inline sliceDomainByInstances(d::MatricialDataset{T,4}, inds::AbstractVector{<:Integer}; return_view = false) where T = if return_view @views d[:, :, inds, :] else d[:, inds, :] end # N=2

@inline strip_domain(d::MatricialDataset{T,2}) where T = d  # N=0
@inline strip_domain(d::MatricialDataset{T,3}) where T = dropdims(d; dims=1)      # N=1
@inline strip_domain(d::MatricialDataset{T,4}) where T = dropdims(d; dims=(1,2))  # N=2

# Initialize MatricialUniDataset by slicing across the features dimension
MatricialUniDataset(::UndefInitializer, d::MatricialDataset{T,2}) where T = Array{T, 1}(undef, n_samples(d))::MatricialUniDataset{T, 1}
MatricialUniDataset(::UndefInitializer, d::MatricialDataset{T,3}) where T = Array{T, 2}(undef, size(d)[1:end-1])::MatricialUniDataset{T, 2}
MatricialUniDataset(::UndefInitializer, d::MatricialDataset{T,4}) where T = Array{T, 3}(undef, size(d)[1:end-1])::MatricialUniDataset{T, 3}

# TODO generalize as init_Xf(X::OntologicalDataset{T, N}) where T = Array{T, N+1}(undef, size(X)[3:end]..., n_samples(X))
@computed struct OntologicalDataset{T,N}
	ontology  :: Ontology
	domain    :: MatricialDataset{T,N+1+1}

	# OntologicalDataset(ontology, domain) = begin
	# 	if prod(channel_size(domain)) == 1
	# 		ontology = ModalLogic.strip_relations(ontology)
	# 	end
	# 	new(ontology, domain)
	# end

end

size(X::OntologicalDataset{T,N})             where {T,N} = size(X.domain)
size(X::OntologicalDataset{T,N}, i::Integer) where {T,N} = size(X.domain, i)
n_samples(X::OntologicalDataset{T,N})        where {T,N} = n_samples(X.domain)
n_variables(X::OntologicalDataset{T,N})      where {T,N} = n_variables(X.domain)
channel_size(X::OntologicalDataset{T,N})     where {T,N} = channel_size(X.domain)


# TODO use Xf[i,[(:) for i in 1:N]...]
# @computed @inline getFeature(X::OntologicalDataset{T,N}, idxs::AbstractVector{Integer}, feature::Integer) where T = X[idxs, feature, fill(:, N)...]::AbstractArray{T,N-1}

# TODO maybe using views can improve performances
# featureview(X::OntologicalDataset{T,0}, idxs::AbstractVector{Integer}, feature::Integer) = X.domain[idxs, feature]
# featureview(X::OntologicalDataset{T,1}, idxs::AbstractVector{Integer}, feature::Integer) = view(X.domain, idxs, feature, :)
# featureview(X::OntologicalDataset{T,2}, idxs::AbstractVector{Integer}, feature::Integer) = view(X.domain, idxs, feature, :, :)

################################################################################
# END Matricial dataset & Ontological dataset
################################################################################

# World generators/enumerators and array/set-like structures
# TODO test the functions for WorldSets with Sets and Arrays, and find the performance optimum
const AbstractWorldSet{W} = Union{AbstractVector{W},AbstractSet{W}} where {W<:AbstractWorld}
# Concrete type for sets: vectors are faster than sets, so we
# const WorldSet = AbstractSet{W} where W<:AbstractWorld
const WorldSet{W} = Vector{W} where {W<:AbstractWorld}
WorldSet{W}(S::WorldSet{W}) where {W<:AbstractWorld} = S

################################################################################
# BEGIN Test operators
################################################################################

abstract type TestOperator end
abstract type TestOperatorPositive <: TestOperator end
abstract type TestOperatorNegative <: TestOperator end

polarity(::TestOperatorPositive) = true
polarity(::TestOperatorNegative) = false

@inline bottom(::TestOperatorPositive, T::Type) = typemin(T)
@inline bottom(::TestOperatorNegative, T::Type) = typemax(T)

@inline opt(::TestOperatorPositive) = max
@inline opt(::TestOperatorNegative) = min

# Warning: I'm assuming all operators are "closed" (= not strict, like >= and <=)
@inline evaluateThreshCondition(::TestOperatorPositive, t::T, gamma::T) where {T} = (t <= gamma)
@inline evaluateThreshCondition(::TestOperatorNegative, t::T, gamma::T) where {T} = (t >= gamma)

struct _TestOpNone  <: TestOperator end; const TestOpNone  = _TestOpNone();
# >=
struct _TestOpGeq  <: TestOperatorPositive end; const TestOpGeq  = _TestOpGeq();
# <=
struct _TestOpLeq  <: TestOperatorNegative end; const TestOpLeq  = _TestOpLeq();

dual_test_operator(::_TestOpGeq) = TestOpLeq
dual_test_operator(::_TestOpLeq) = TestOpGeq

primary_test_operator(x::_TestOpGeq) = TestOpGeq # x
primary_test_operator(x::_TestOpLeq) = TestOpGeq # dual_test_operator(x)

siblings(::_TestOpGeq) = []
siblings(::_TestOpLeq) = []

# >=_α
struct _TestOpGeqSoft  <: TestOperatorPositive
  alpha :: AbstractFloat
  _TestOpGeqSoft(a::T) where {T<:Real} = (a > 0 && a < 1) ? new(a) : error("Invalid instantiation for test operator: _TestOpGeqSoft($(a))")
end;
const TestOpGeq_95  = _TestOpGeqSoft((Rational(95,100)));
const TestOpGeq_90  = _TestOpGeqSoft((Rational(90,100)));
const TestOpGeq_85  = _TestOpGeqSoft((Rational(85,100)));
const TestOpGeq_80  = _TestOpGeqSoft((Rational(80,100)));
const TestOpGeq_75  = _TestOpGeqSoft((Rational(75,100)));
const TestOpGeq_70  = _TestOpGeqSoft((Rational(70,100)));
const TestOpGeq_60  = _TestOpGeqSoft((Rational(60,100)));

# <_α
struct _TestOpLeqSoft  <: TestOperatorNegative
  alpha :: AbstractFloat
  _TestOpLeqSoft(a::T) where {T<:Real} = (a > 0 && a < 1) ? new(a) : error("Invalid instantiation for test operator: _TestOpLeqSoft($(a))")
end;
const TestOpLeq_95  = _TestOpLeqSoft((Rational(95,100)));
const TestOpLeq_90  = _TestOpLeqSoft((Rational(90,100)));
const TestOpLeq_85  = _TestOpLeqSoft((Rational(85,100)));
const TestOpLeq_80  = _TestOpLeqSoft((Rational(80,100)));
const TestOpLeq_75  = _TestOpLeqSoft((Rational(75,100)));
const TestOpLeq_70  = _TestOpLeqSoft((Rational(70,100)));
const TestOpLeq_60  = _TestOpLeqSoft((Rational(60,100)));

alpha(x::_TestOpGeqSoft) = x.alpha
alpha(x::_TestOpLeqSoft) = x.alpha

dual_test_operator(x::_TestOpGeqSoft) = TestOpNone
dual_test_operator(x::_TestOpLeqSoft) = TestOpNone
# TODO fix
# dual_test_operator(x::_TestOpGeqSoft) = error("If you use $(x), need to write computeModalThresholdDual for the primal test operator.")
# dual_test_operator(x::_TestOpLeqSoft) = error("If you use $(x), need to write computeModalThresholdDual for the primal test operator.")

primary_test_operator(x::_TestOpGeqSoft) = x
primary_test_operator(x::_TestOpLeqSoft) = dual_test_operator(x)

const SoftenedOperators = [
											TestOpGeq_95, TestOpLeq_95,
											TestOpGeq_90, TestOpLeq_90,
											TestOpGeq_80, TestOpLeq_80,
											TestOpGeq_85, TestOpLeq_85,
											TestOpGeq_75, TestOpLeq_75,
											TestOpGeq_70, TestOpLeq_70,
											TestOpGeq_60, TestOpLeq_60,
										]

siblings(x::Union{_TestOpGeqSoft,_TestOpLeqSoft}) = SoftenedOperators

const all_lowlevel_test_operators = [
		TestOpGeq, TestOpLeq,
		SoftenedOperators...
	]

const all_ordered_test_operators = [
		TestOpGeq, TestOpLeq,
		SoftenedOperators...
	]
const all_test_operators_order = [
		TestOpGeq, TestOpLeq,
		SoftenedOperators...
	]
sort_test_operators!(x::Vector{TO}) where {TO<:TestOperator} = begin
	intersect(all_test_operators_order, x)
end

display_test_operator(test_operator::_TestOpGeq) = "⫺"
display_test_operator(test_operator::_TestOpLeq) = "⫹"
display_test_operator(test_operator::_TestOpGeqSoft) = "⫺" * subscriptnumber(rstrip(rstrip(string(alpha(test_operator)*100), '0'), '.'))
display_test_operator(test_operator::_TestOpLeqSoft) = "⫹" * subscriptnumber(rstrip(rstrip(string(alpha(test_operator)*100), '0'), '.'))

display_propositional_test(test_operator::_TestOpGeq, lhs::String, featval::Number) = "$(lhs) ⫺ $(featval)"
display_propositional_test(test_operator::_TestOpLeq, lhs::String, featval::Number) = "$(lhs) ⫹ $(featval)"
display_propositional_test(test_operator::_TestOpGeqSoft, lhs::String, featval::Number) = "$(lhs) ⫺" * subscriptnumber(rstrip(rstrip(string(alpha(test_operator)*100), '0'), '.')) * " $(featval)"
display_propositional_test(test_operator::_TestOpLeqSoft, lhs::String, featval::Number) = "$(lhs) ⫹" * subscriptnumber(rstrip(rstrip(string(alpha(test_operator)*100), '0'), '.')) * " $(featval)"
# display_propositional_test(test_operator::_TestOpGeqSoft, lhs::String, featval::Number) = "$(alpha(test_operator)*100)% [$(lhs) ⫺ $(featval)]"
# display_propositional_test(test_operator::_TestOpLeqSoft, lhs::String, featval::Number) = "$(alpha(test_operator)*100)% [$(lhs) ⫹ $(featval)]"

display_modal_test(modality::AbstractRelation, test_operator::TestOperator, featid::Integer, featval::Number) = begin
	test = display_propositional_test(test_operator, "V$(featid)", featval)
	if modality != RelationId
		"$(display_existential_modality(modality)) ($test)"
	else
		"$test"
	end
end

@inline computePropositionalThresholdDual(::_TestOpGeq, w::AbstractWorld, channel::MatricialChannel{T,N}) where {T,N} = extrema(readWorld(w,channel))
@inline computePropositionalThreshold(::_TestOpGeq, w::AbstractWorld, channel::MatricialChannel{T,N}) where {T,N} = begin
	# println(_TestOpGeq)
	# println(w)
	# println(channel)
	# println(maximum(readWorld(w,channel)))
	# readline()
	minimum(readWorld(w,channel))
end
@inline computePropositionalThreshold(::_TestOpLeq, w::AbstractWorld, channel::MatricialChannel{T,N}) where {T,N} = begin
	# println(_TestOpLeq)
	# println(w)
	# println(channel)
	# readline()
	maximum(readWorld(w,channel))
end

# TODO improved version for Rational numbers
# TODO check
@inline test_op_partialsort!(test_op::_TestOpGeqSoft, vals::Vector{T}) where {T} = 
	partialsort!(vals,ceil(Int, alpha(test_op)*length(vals)); rev=true)
@inline test_op_partialsort!(test_op::_TestOpLeqSoft, vals::Vector{T}) where {T} = 
	partialsort!(vals,ceil(Int, alpha(test_op)*length(vals)))

# TODO think about this:
# @inline computePropositionalThresholdDual(test_op::_TestOpGeqSoft, w::AbstractWorld, channel::MatricialChannel{T,N}) where {T,N} = begin
# 	vals = vec(readWorld(w,channel))
# 	xmin = test_op_partialsort!(test_op,vec(readWorld(w,channel)))
# 	xmin = partialsort!(vals,ceil(Int, alpha(test_op)*length(vals)); rev=true)
# 	xmax = partialsort!(vals,ceil(Int, (alpha(test_op))*length(vals)))
# 	xmin,xmax
# end
@inline computePropositionalThreshold(test_op::Union{_TestOpGeqSoft,_TestOpLeqSoft}, w::AbstractWorld, channel::MatricialChannel{T,N}) where {T,N} = begin
	vals = vec(readWorld(w,channel))
	test_op_partialsort!(test_op,vals)
end
@inline computePropositionalThresholdMany(test_ops::Vector{<:TestOperator}, w::AbstractWorld, channel::MatricialChannel{T,N}) where {T,N} = begin
	vals = vec(readWorld(w,channel))
	(test_op_partialsort!(test_op,vals) for test_op in test_ops)
end

computeModalThresholdDual(test_operator::TestOperatorPositive, w::WorldType, relation::AbstractRelation, channel::MatricialChannel{T,N}) where {WorldType<:AbstractWorld,T,N} = begin
	worlds = enumAccessibles([w], relation, channel)
	extr = (typemin(T),typemax(T))
	for w in worlds
		e = computePropositionalThresholdDual(test_operator, w, channel)
		extr = (min(extr[1],e[1]), max(extr[2],e[2]))
	end
	extr
end
# TODO write a single computeModalThreshold using bottom and opt
# TODO use readGammas
computeModalThreshold(test_operator::TestOperatorPositive, w::WorldType, relation::AbstractRelation, channel::MatricialChannel{T,N}) where {WorldType<:AbstractWorld,T,N} = begin
	worlds = enumAccessibles([w], relation, channel)
	v = typemin(T) # TODO write with reduce
	for w in worlds
		e = computePropositionalThreshold(test_operator, w, channel)
		v = max(v,e)
	end
	v
end
computeModalThreshold(test_operator::TestOperatorNegative, w::WorldType, relation::AbstractRelation, channel::MatricialChannel{T,N}) where {WorldType<:AbstractWorld,T,N} = begin
	worlds = enumAccessibles([w], relation, channel)
	v = typemax(T) # TODO write with reduce
	for w in worlds
		e = computePropositionalThreshold(test_operator, w, channel)
		v = min(v,e)
	end
	v
end

computeModalThresholdMany(test_ops::Vector{<:TestOperator}, w::WorldType, relation::AbstractRelation, channel::MatricialChannel{T,N}) where {WorldType<:AbstractWorld,T,N} = begin
	[computeModalThreshold(test_op, w, relation, channel) for test_op in test_ops]
end

# TODO remove
# @inline WMax(w::AbstractWorld, channel::MatricialChannel{T,N}) where {T,N} = maximum(readWorld(w,channel))
# @inline WMin(w::AbstractWorld, channel::MatricialChannel{T,N}) where {T,N} = minimum(readWorld(w,channel))

@inline testCondition(test_operator::_TestOpGeq, w::AbstractWorld, channel::MatricialChannel{T,N}, featval::Number) where {T,N} = begin # TODO maybe this becomes SIMD, or sum/all(readWorld(w,channel)  .<= featval)
	# Source: https://stackoverflow.com/questions/47564825/check-if-all-the-elements-of-a-julia-array-are-equal
	# @inbounds
	for x in readWorld(w,channel)
		x >= featval || return false
	end
	return true
end
@inline testCondition(test_operator::_TestOpLeq, w::AbstractWorld, channel::MatricialChannel{T,N}, featval::Number) where {T,N} = begin # TODO maybe this becomes SIMD, or sum/all(readWorld(w,channel)  .<= featval)
	# Source: https://stackoverflow.com/questions/47564825/check-if-all-the-elements-of-a-julia-array-are-equal
	# @info "WLes" w featval #n readWorld(w,channel)
	# @inbounds
	for x in readWorld(w,channel)
		x <= featval || return false
	end
	return true
end

@inline testCondition(test_operator::_TestOpGeqSoft, w::AbstractWorld, channel::MatricialChannel{T,N}, featval::Number) where {T,N} = begin 
	ys = 0
	vals = readWorld(w,channel)
	# print(vals, ": ")
	for x in vals
		if x >= featval
			# print(x, ",")
			ys+=1
		end
	end
	# println()
	# println(ys, (ys/length(vals)), (ys/length(vals)) >= test_operator.alpha)
	(ys/length(vals)) >= test_operator.alpha
end

@inline testCondition(test_operator::_TestOpLeqSoft, w::AbstractWorld, channel::MatricialChannel{T,N}, featval::Number) where {T,N} = begin 
	ys = 0
	vals = readWorld(w,channel)
	for x in vals
		if x <= featval
			ys+=1
		end
	end
	(ys/length(vals)) >= test_operator.alpha
end

# Utility type for enhanced computation of thresholds
abstract type _ReprTreatment end
struct _ReprFake{worldType<:AbstractWorld}  <: _ReprTreatment w :: worldType end
struct _ReprMax{worldType<:AbstractWorld}  <: _ReprTreatment w :: worldType end
struct _ReprMin{worldType<:AbstractWorld}  <: _ReprTreatment w :: worldType end
struct _ReprVal{worldType<:AbstractWorld}  <: _ReprTreatment w :: worldType end
struct _ReprNone{worldType<:AbstractWorld} <: _ReprTreatment end

################################################################################
# END Test operators
################################################################################

## Enumerate accessible worlds

# Fallback: enumAccessibles works with domains AND their dimensions
enumAccessibles(S::Any, r::AbstractRelation, channel::MatricialChannel{T,N}) where {T,N} = enumAccessibles(S, r, size(channel)...)
enumAccRepr(S::Any, r::AbstractRelation, channel::MatricialChannel{T,N}) where {T,N} = enumAccRepr(S, r, size(channel)...)
# Fallback: enumAccessibles for world sets maps to enumAccessibles-ing their elements
#  (note: one may overload this function to provide improved implementations for special cases (e.g. <L> of a world set in interval algebra))
enumAccessibles(S::AbstractWorldSet{WorldType}, r::AbstractRelation, XYZ::Vararg{Integer,N}) where {T,N,WorldType<:AbstractWorld} = begin
	IterTools.imap(WorldType,
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

enumAccessibles(w::WorldType,           ::_RelationId, XYZ::Vararg{Integer,N}) where {WorldType<:AbstractWorld,N} = [w]
enumAccessibles(S::AbstractWorldSet{W}, ::_RelationId, XYZ::Vararg{Integer,N}) where {W<:AbstractWorld,N} = S # TODO try IterTools.imap(identity, S) ?
# Maybe this will have a use: enumAccessiblesW1(w::AbstractWorld, ::_RelationId,   X::Integer) where T = [w] # IterTools.imap(identity, [w])

# TODO parametrize on test operator (any test operator in this case)
enumAccRepr(w::WorldType, ::_RelationId, XYZ::Vararg{Integer,N}) where {WorldType<:AbstractWorld,N} = [w]
computeModalThresholdDual(test_operator::TestOperator, w::WorldType, relation::_RelationId, channel::MatricialChannel{T,N}) where {WorldType<:AbstractWorld,T,N} =
	computePropositionalThresholdDual(test_operator, w, channel)
computeModalThreshold(test_operator::TestOperator, w::WorldType, relation::_RelationId, channel::MatricialChannel{T,N}) where {WorldType<:AbstractWorld,T,N} =
	computePropositionalThreshold(test_operator, w, channel)

display_rel_short(::_RelationId)  = "Id"
display_rel_short(::_RelationAll) = ""

# TODO
# A relation can be defined as a union of other relations.
# In this case, thresholds can be computed by maximization/minimization of the
#  thresholds referred to the relations involved.
# abstract type AbstractRelation end
struct _UnionOfRelations{T<:NTuple{N,<:AbstractRelation} where N} <: AbstractRelation end;

# computeModalThresholdDual(test_operator::TestOperator, w::WorldType, relation::R where R<:_UnionOfRelations{relsTuple}, channel::MatricialChannel{T,N}) where {WorldType<:AbstractWorld,T,N} =
# 	computePropositionalThresholdDual(test_operator, w, channel)
# 	fieldtypes(relsTuple)
# computeModalThreshold(test_operator::TestOperator, w::WorldType, relation::R where R<:_UnionOfRelations{relsTuple}, channel::MatricialChannel{T,N}) where {WorldType<:AbstractWorld,T,N} =
# 	computePropositionalThreshold(test_operator, w, channel)
# 	fieldtypes(relsTuple)


# Perform the modal step, that is, evaluate a modal formula
#  on a domain, and eventually compute the new world set.
# TODO perhaps fastMode never needed, figure out
modalStep(S::WorldSetType,
					relation::R,
					channel::AbstractArray{T,N},
					test_operator::TestOperator,
					threshold::T) where {W<:AbstractWorld, WorldSetType<:Union{AbstractSet{W},AbstractVector{W}}, R<:AbstractRelation, T, N} = begin
	@logmsg DTDetail "modalStep" S relation display_modal_test(relation, test_operator, -1, threshold)
	satisfied = false
	worlds = enumAccessibles(S, relation, channel)
	if length(collect(Iterators.take(worlds, 1))) > 0
		new_worlds = WorldSetType()
		for w in worlds
			if testCondition(test_operator, w, channel, threshold)
				@logmsg DTDetail " Found world " w readWorld(w,channel)
				satisfied = true
				push!(new_worlds, w)
			end
		end
		if satisfied == true
			S = new_worlds
		else 
			# If none of the neighboring worlds satisfies the condition, then 
			#  the new set is left unchanged
		end
	else
		@logmsg DTDetail "   No world found"
		# If there are no neighboring worlds, then the modal condition is not met
	end
	if satisfied
		@logmsg DTDetail "   YES" S
	else
		@logmsg DTDetail "   NO" 
	end
	return (satisfied,S)
end

################################################################################
################################################################################
################################################################################

show(io::IO, r::AbstractRelation) = print(io, display_existential_modality(r))
display_existential_modality(r) = "⟨" * display_rel_short(r) * "⟩"

# Note: with under 10 values, computation on tuples is faster
# xtup = (zip(randn(2),randn(2)) |> collect |> Tuple);
# xarr = (zip(randn(2),randn(2)) |> collect);
# @btime min3Extrema($xtup)
# @btime min3Extrema($xarr)
# xtup = (zip(randn(10),randn(10)) |> collect |> Tuple);
# xarr = (zip(randn(10),randn(10)) |> collect);
# @btime min3Extrema($xtup)
# @btime min3Extrema($xarr)
# xtup = (zip(randn(1000),randn(1000)) |> collect |> Tuple);
# xarr = (zip(randn(1000),randn(1000)) |> collect);
# @btime min3Extrema($xtup)
# @btime min3Extrema($xarr)
minExtrema(extr::Union{NTuple{N,NTuple{2,T}},AbstractVector{NTuple{2,T}}}) where {T<:Number,N} = reduce(((fst,snd),(f,s))->(min(fst,f),max(snd,s)), extr; init=(typemax(T),typemin(T)))
maxExtrema(extr::Union{NTuple{N,NTuple{2,T}},AbstractVector{NTuple{2,T}}}) where {T<:Number,N} = reduce(((fst,snd),(f,s))->(max(fst,f),min(snd,s)), extr; init=(typemin(T),typemax(T)))
minExtrema(extr::Vararg{NTuple{2,T}}) where {T<:Number} = minExtrema(extr)
maxExtrema(extr::Vararg{NTuple{2,T}}) where {T<:Number} = maxExtrema(extr)

include("Interval.jl")
include("IARelations.jl")
include("TopoRelations.jl")
include("Interval2D.jl")
include("IA2DRelations.jl")
include("Topo2DRelations.jl")
# include("Point.jl")


export genericIntervalOntology,
				IntervalOntology,
				Interval2DOntology,
				getIntervalOntologyOfDim,
				genericIntervalRCC8Ontology,
				IntervalRCC8Ontology,
				Interval2DRCC8Ontology,
				getIntervalRCC8OntologyOfDim,
				getIntervalRCC5OntologyOfDim,
				IntervalRCC5Ontology,
				Interval2DRCC5Ontology

abstract type OntologyType end
struct _genericIntervalOntology  <: OntologyType end; const genericIntervalOntology  = _genericIntervalOntology();  # After
const IntervalOntology   = Ontology(Interval,IARelations)
const Interval2DOntology = Ontology(Interval2D,IA2DRelations)

struct _genericIntervalRCC8Ontology  <: OntologyType end; const genericIntervalRCC8Ontology  = _genericIntervalRCC8Ontology();  # After
const IntervalRCC8Ontology   = Ontology(Interval,RCC8Relations)
const Interval2DRCC8Ontology = Ontology(Interval2D,RCC8Relations)
const IntervalRCC5Ontology   = Ontology(Interval,RCC5Relations)
const Interval2DRCC5Ontology = Ontology(Interval2D,RCC5Relations)
 
getIntervalOntologyOfDim(::MatricialDataset{T,D}) where {T,D} = getIntervalOntologyOfDim(Val(D-2))
getIntervalOntologyOfDim(::Val{1}) = IntervalOntology
getIntervalOntologyOfDim(::Val{2}) = Interval2DOntology

getIntervalRCC8OntologyOfDim(::MatricialDataset{T,D}) where {T,D} = getIntervalRCC8OntologyOfDim(Val(D-2))
getIntervalRCC8OntologyOfDim(::Val{1}) = IntervalRCC8Ontology
getIntervalRCC8OntologyOfDim(::Val{2}) = Interval2DRCC8Ontology

getIntervalRCC5OntologyOfDim(::MatricialDataset{T,D}) where {T,D} = getIntervalRCC5OntologyOfDim(Val(D-2))
getIntervalRCC5OntologyOfDim(::Val{1}) = IntervalRCC5Ontology
getIntervalRCC5OntologyOfDim(::Val{2}) = Interval2DRCC5Ontology

end # module
