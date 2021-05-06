export TestOperator,
				TestOpNone,
				TestOpGeq, TestOpLeq,
				_TestOpGeqSoft, _TestOpLeqSoft

abstract type TestOperator end

################################################################################
################################################################################

struct _TestOpNone  <: TestOperator end; const TestOpNone  = _TestOpNone();

################################################################################
################################################################################

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

computeModalThreshold(test_operator::Union{TestOperatorPositive,TestOperatorNegative}, w::WorldType, relation::AbstractRelation, channel::MatricialChannel{T,N}) where {WorldType<:AbstractWorld,T,N} = begin
	worlds = enumAccessibles([w], relation, channel)
	# TODO rewrite as reduce(opt(test_operator), (computePropositionalThreshold(test_operator, w, channel) for w in worlds); init=bottom(test_operator, T))
	v = bottom(test_operator, T)
	for w in worlds
		e = computePropositionalThreshold(test_operator, w, channel)
		v = opt(test_operator)(v,e)
	end
	v
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
computeModalThresholdMany(test_ops::Vector{<:TestOperator}, w::WorldType, relation::AbstractRelation, channel::MatricialChannel{T,N}) where {WorldType<:AbstractWorld,T,N} = begin
	[computeModalThreshold(test_op, w, relation, channel) for test_op in test_ops]
end

################################################################################
################################################################################

# ⫺ and ⫹, that is, "*all* of the values on this world are at least, or at most ..."
struct _TestOpGeq  <: TestOperatorPositive end; const TestOpGeq  = _TestOpGeq();
struct _TestOpLeq  <: TestOperatorNegative end; const TestOpLeq  = _TestOpLeq();

dual_test_operator(::_TestOpGeq) = TestOpLeq
dual_test_operator(::_TestOpLeq) = TestOpGeq

# TODO introduce singleton design pattern for these constants
primary_test_operator(x::_TestOpGeq) = TestOpGeq # x
primary_test_operator(x::_TestOpLeq) = TestOpGeq # dual_test_operator(x)

siblings(::_TestOpGeq) = []
siblings(::_TestOpLeq) = []

Base.show(io::IO, test_operator::_TestOpGeq) = print(io, "⫺")
Base.show(io::IO, test_operator::_TestOpLeq) = print(io, "⫹")

@inline computePropositionalThreshold(::_TestOpGeq, w::AbstractWorld, channel::MatricialChannel{T,N}) where {T,N} = begin
	# println(_TestOpGeq)
	# println(w)
	# println(channel)
	# println(maximum(readWorld(w,channel)))
	# readline()
	minimum(readWorld(w,channel))
end
@inline computePropositionalThresholdDual(::_TestOpGeq, w::AbstractWorld, channel::MatricialChannel{T,N}) where {T,N} = extrema(readWorld(w,channel))
@inline computePropositionalThreshold(::_TestOpLeq, w::AbstractWorld, channel::MatricialChannel{T,N}) where {T,N} = begin
	# println(_TestOpLeq)
	# println(w)
	# println(channel)
	# readline()
	maximum(readWorld(w,channel))
end

@inline testCondition(test_operator::_TestOpGeq, w::AbstractWorld, channel::MatricialChannel{T,N}, featval::Number) where {T,N} = begin # TODO maybe this becomes SIMD, or sum/all(readWorld(w,channel)  .<= featval)
	# Source: https://stackoverflow.com/questions/47564825/check-if-all-the-elements-of-a-julia-array-are-equal
	# @inbounds
	# TODO try:
	# all(readWorld(w,channel) .>= featval)
	for x in readWorld(w,channel)
		x >= featval || return false
	end
	return true
end
@inline testCondition(test_operator::_TestOpLeq, w::AbstractWorld, channel::MatricialChannel{T,N}, featval::Number) where {T,N} = begin # TODO maybe this becomes SIMD, or sum/all(readWorld(w,channel)  .<= featval)
	# Source: https://stackoverflow.com/questions/47564825/check-if-all-the-elements-of-a-julia-array-are-equal
	# @info "WLes" w featval #n readWorld(w,channel)
	# @inbounds
	# TODO try:
	# all(readWorld(w,channel) .<= featval)
	for x in readWorld(w,channel)
		x <= featval || return false
	end
	return true
end

################################################################################
################################################################################

export TestOpGeq_95, TestOpGeq_90, TestOpGeq_85, TestOpGeq_80, TestOpGeq_75, TestOpGeq_70, TestOpGeq_60,
				TestOpLeq_95, TestOpLeq_90, TestOpLeq_85, TestOpLeq_80, TestOpLeq_75, TestOpLeq_70, TestOpLeq_60

# ⫺_α and ⫹_α, that is, "*at least α⋅100 percent* of the values on this world are at least, or at most ..."

struct _TestOpGeqSoft  <: TestOperatorPositive
  alpha :: AbstractFloat
  _TestOpGeqSoft(a::T) where {T<:Real} = (a > 0 && a < 1) ? new(a) : error("Invalid instantiation for test operator: _TestOpGeqSoft($(a))")
end;
struct _TestOpLeqSoft  <: TestOperatorNegative
  alpha :: AbstractFloat
  _TestOpLeqSoft(a::T) where {T<:Real} = (a > 0 && a < 1) ? new(a) : error("Invalid instantiation for test operator: _TestOpLeqSoft($(a))")
end;

const TestOpGeq_95  = _TestOpGeqSoft((Rational(95,100)));
const TestOpGeq_90  = _TestOpGeqSoft((Rational(90,100)));
const TestOpGeq_85  = _TestOpGeqSoft((Rational(85,100)));
const TestOpGeq_80  = _TestOpGeqSoft((Rational(80,100)));
const TestOpGeq_75  = _TestOpGeqSoft((Rational(75,100)));
const TestOpGeq_70  = _TestOpGeqSoft((Rational(70,100)));
const TestOpGeq_60  = _TestOpGeqSoft((Rational(60,100)));

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
# TODO The dual_test_operators for TestOpGeqSoft(alpha) is TestOpLeSoft(1-alpha), which is not defined yet.
# Define it, together with their dual_test_operator and computePropositionalThresholdDual
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

Base.show(io::IO, test_operator::_TestOpGeqSoft) = print(io, "⫺" * subscriptnumber(rstrip(rstrip(string(alpha(test_operator)*100), '0'), '.')))
Base.show(io::IO, test_operator::_TestOpLeqSoft) = print(io, "⫹" * subscriptnumber(rstrip(rstrip(string(alpha(test_operator)*100), '0'), '.')))

# TODO improved version for Rational numbers
# TODO check
@inline test_op_partialsort!(test_op::_TestOpGeqSoft, vals::Vector{T}) where {T} = 
	partialsort!(vals,ceil(Int, alpha(test_op)*length(vals)); rev=true)
@inline test_op_partialsort!(test_op::_TestOpLeqSoft, vals::Vector{T}) where {T} = 
	partialsort!(vals,ceil(Int, alpha(test_op)*length(vals)))

@inline computePropositionalThreshold(test_op::Union{_TestOpGeqSoft,_TestOpLeqSoft}, w::AbstractWorld, channel::MatricialChannel{T,N}) where {T,N} = begin
	vals = vec(readWorld(w,channel))
	test_op_partialsort!(test_op,vals)
end
# @inline computePropositionalThresholdDual(test_op::_TestOpGeqSoft, w::AbstractWorld, channel::MatricialChannel{T,N}) where {T,N} = begin
# 	vals = vec(readWorld(w,channel))
# 	xmin = test_op_partialsort!(test_op,vec(readWorld(w,channel)))
# 	xmin = partialsort!(vals,ceil(Int, alpha(test_op)*length(vals)); rev=true)
# 	xmax = partialsort!(vals,ceil(Int, (alpha(test_op))*length(vals)))
# 	xmin,xmax
# end
@inline computePropositionalThresholdMany(test_ops::Vector{<:TestOperator}, w::AbstractWorld, channel::MatricialChannel{T,N}) where {T,N} = begin
	vals = vec(readWorld(w,channel))
	(test_op_partialsort!(test_op,vals) for test_op in test_ops)
end

@inline testCondition(test_operator::_TestOpGeqSoft, w::AbstractWorld, channel::MatricialChannel{T,N}, featval::Number) where {T,N} = begin 
	ys = 0
	# TODO write with reduce, and optimize it (e.g. by stopping early if the condition is reached already)
	vals = readWorld(w,channel)
	for x in vals
		if x >= featval
			ys+=1
		end
	end
	(ys/length(vals)) >= test_operator.alpha
end

@inline testCondition(test_operator::_TestOpLeqSoft, w::AbstractWorld, channel::MatricialChannel{T,N}, featval::Number) where {T,N} = begin 
	ys = 0
	# TODO write with reduce, and optimize it (e.g. by stopping early if the condition is reached already)
	vals = readWorld(w,channel)
	for x in vals
		if x <= featval
			ys+=1
		end
	end
	(ys/length(vals)) >= test_operator.alpha
end

################################################################################
################################################################################


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
