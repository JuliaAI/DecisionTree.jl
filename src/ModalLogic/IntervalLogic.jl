export # Interval, x, y,
				# readMax, readMin,
				IntervalOntology

# Interval
struct Interval <: AbstractWorld
	x :: Integer
	y :: Integer
	# TODO check x<y but only in debug mode.  && x<=N, y<=N
	# Interval(x,y) = x>0 && y>0 && x < y ? new(x,y) : error("Can't instantiate (x={$x},y={$y})")
	Interval(x,y) = new(x,y)
	Interval(::_emptyWorld) = new(-1,0)
	Interval(::_centeredWorld, X::Integer) = new(div(X,2)+1,div(X,2)+1+1+(isodd(X) ? 0 : 1))
end

convert(::Type{Interval}, t::Tuple{Integer,Integer}) where {S, T} = Interval(t...)
Interval(params::Tuple{Integer,Integer}) = Interval(params...)

x(w::Interval) = w.x
y(w::Interval) = w.y

# map((x)->readWorld(Interval(x),[1,2,3,4,5]), enumIntervalsInRange(1,6) |> collect)
@inline readWorld(w::Interval, channel::MatricialChannel{T,1}) where {T} = @inbounds channel[w.x:w.y-1]

# 6+6 Interval relations
abstract type _IARelation <: AbstractRelation end
const _IA_I = _RelationId; const IA_I = RelationId
const _IA_U = _RelationAll; const IA_U = RelationAll
struct _IA_A  <: _IARelation end; const IA_A  = _IA_A(); # After
struct _IA_L  <: _IARelation end; const IA_L  = _IA_L(); # Later
struct _IA_B  <: _IARelation end; const IA_B  = _IA_B(); # Begins
struct _IA_E  <: _IARelation end; const IA_E  = _IA_E(); # Ends
struct _IA_D  <: _IARelation end; const IA_D  = _IA_D(); # During
struct _IA_O  <: _IARelation end; const IA_O  = _IA_O(); # Overlaps
struct _IA_Ai <: _IARelation end; const IA_Ai = _IA_Ai(); # inverse(After)
struct _IA_Li <: _IARelation end; const IA_Li = _IA_Li(); # inverse(Later)
struct _IA_Bi <: _IARelation end; const IA_Bi = _IA_Bi(); # inverse(Begins)
struct _IA_Ei <: _IARelation end; const IA_Ei = _IA_Ei(); # inverse(Ends)
struct _IA_Di <: _IARelation end; const IA_Di = _IA_Di(); # inverse(During)
struct _IA_Oi <: _IARelation end; const IA_Oi = _IA_Oi(); # inverse(Overlaps)

print_rel_short(::_IA_A)  = "A"
print_rel_short(::_IA_L)  = "L"
print_rel_short(::_IA_B)  = "B"
print_rel_short(::_IA_E)  = "E"
print_rel_short(::_IA_D)  = "D"
print_rel_short(::_IA_O)  = "O"
print_rel_short(::_IA_Ai) = "Ai"
print_rel_short(::_IA_Li) = "Li"
print_rel_short(::_IA_Bi) = "Bi"
print_rel_short(::_IA_Ei) = "Ei"
print_rel_short(::_IA_Di) = "Di"
print_rel_short(::_IA_Oi) = "Oi"

# 12 Interval Algebra relations
const IARelations = [
IA_A,  IA_L,  IA_B,  IA_E,  IA_D,  IA_O,
IA_Ai, IA_Li, IA_Bi, IA_Ei, IA_Di, IA_Oi
]

# 13 Interval Algebra extended relations (IA extended with universal)
const IARelations_extended = [RelationAll,IARelations]

# Thought:
#  Domanda: ci serve iterare su generatori o no?
#  Nel learning filtro i mondi a seconda di quali soddisfano le clausole.
#  Posso farlo usando generatori, chissa', forse e' piu' conveniente?
#  Nel frattempo preparo il codice sia per generatori che per arrays

# Enumerate intervals in a given range
enumIntervalsInRange(a::Integer, b::Integer) =
	Iterators.filter((a)->a[1]<a[2], Iterators.product(a:b-1, a+1:b))

# enumAccW1(w::Interval, ::_RelationAll,    X::Integer) where T =
	# IterTools.imap(Interval, enumIntervalsInRange(1, X+1))

## Enumerate accessible worlds from a single world
enumAccW1(w::Interval, ::_IA_A,    X::Integer) where T =
	IterTools.imap((y)->Interval(w.y, y), w.y+1:X+1)
enumAccW1(w::Interval, ::_IA_Ai,   X::Integer) where T =
	IterTools.imap((x)->Interval(x, w.x), 1:w.x-1)
enumAccW1(w::Interval, ::_IA_L,    X::Integer) where T =
	IterTools.imap(Interval, enumIntervalsInRange(w.y+1, X+1))
enumAccW1(w::Interval, ::_IA_Li,   X::Integer) where T =
	IterTools.imap(Interval, enumIntervalsInRange(1, w.x-1))
enumAccW1(w::Interval, ::_IA_B,    X::Integer) where T =
	IterTools.imap((y)->Interval(w.x, y), w.x+1:w.y-1)
enumAccW1(w::Interval, ::_IA_Bi,   X::Integer) where T =
	IterTools.imap((y)->Interval(w.x, y), w.y+1:X+1)
enumAccW1(w::Interval, ::_IA_E,    X::Integer) where T =
	IterTools.imap((x)->Interval(x, w.y), w.x+1:w.y-1)
enumAccW1(w::Interval, ::_IA_Ei,   X::Integer) where T =
	IterTools.imap((x)->Interval(x, w.y), 1:w.x-1)
enumAccW1(w::Interval, ::_IA_D,    X::Integer) where T =
	IterTools.imap(Interval, enumIntervalsInRange(w.x+1, w.y-1))
enumAccW1(w::Interval, ::_IA_Di,   X::Integer) where T =
	IterTools.imap(Interval, Iterators.product(1:w.x-1, w.y+1:X+1))
enumAccW1(w::Interval, ::_IA_O,    X::Integer) where T =
	IterTools.imap(Interval, Iterators.product(w.x+1:w.y-1, w.y+1:X+1))
enumAccW1(w::Interval, ::_IA_Oi,   X::Integer) where T =
	IterTools.imap(Interval, Iterators.product(1:w.x-1, w.x+1:w.y-1))

## Enumerate accessible worlds from a set of worlds
enumAcc1(S::Union{WorldGenerator,AbstractWorldSet{Interval}}, r::R where R<:_IARelation, X::Integer) where T = begin
	IterTools.distinct(Iterators.flatten((enumAccW1(w, r, X) for w in S)))
end

# More efficient implementations for edge cases

enumAcc1_1(S::Union{WorldGenerator,AbstractWorldSet{Interval}}, ::_IA_L, X::Integer) where T = begin
	# @show Base.argmin((w.y for w in S))
	enumAccW1(Base.argmin((w.y for w in S)), IA_L, X)
end
enumAcc1_1(S::Union{WorldGenerator,AbstractWorldSet{Interval}}, ::_IA_Li, X::Integer) where T = begin
	# @show Base.argmax((w.x for w in S))
	enumAccW1(Base.argmax((w.x for w in S)), IA_Li, X)
end
enumAcc1_2(S::Union{WorldGenerator,AbstractWorldSet{Interval}}, ::_IA_L, X::Integer) where T = 
	enumAccW1(S[argmin(y.(S))], IA_L, X)
enumAcc1_2(S::Union{WorldGenerator,AbstractWorldSet{Interval}}, ::_IA_Li, X::Integer) where T = 
	enumAccW1(S[argmax(x.(S))], IA_Li, X)

#####

# enumAccW2(w::Interval, ::_RelationAll,  X::Integer) where T = enumIntervalsInRange(1, X+1)
## Enumerate accessible worlds from a single world
enumAccW2(w::Interval, ::_IA_A,  X::Integer) where T = zip(Iterators.repeated(w.y), w.y+1:X+1)
enumAccW2(w::Interval, ::_IA_Ai, X::Integer) where T = zip(1:w.x-1, Iterators.repeated(w.x))
enumAccW2(w::Interval, ::_IA_L,  X::Integer) where T = enumIntervalsInRange(w.y+1, X+1)
enumAccW2(w::Interval, ::_IA_Li, X::Integer) where T = enumIntervalsInRange(1, w.x-1)
enumAccW2(w::Interval, ::_IA_B,  X::Integer) where T = zip(Iterators.repeated(w.x), w.x+1:w.y-1)
enumAccW2(w::Interval, ::_IA_Bi, X::Integer) where T = zip(Iterators.repeated(w.x), w.y+1:X+1)
enumAccW2(w::Interval, ::_IA_E,  X::Integer) where T = zip(w.x+1:w.y-1, Iterators.repeated(w.y))
enumAccW2(w::Interval, ::_IA_Ei, X::Integer) where T = zip(1:w.x-1, Iterators.repeated(w.y))
enumAccW2(w::Interval, ::_IA_D,  X::Integer) where T = enumIntervalsInRange(w.x+1, w.y-1)
enumAccW2(w::Interval, ::_IA_Di, X::Integer) where T = Iterators.product(1:w.x-1, w.y+1:X+1)
enumAccW2(w::Interval, ::_IA_O,  X::Integer) where T = Iterators.product(w.x+1:w.y-1, w.y+1:X+1)
enumAccW2(w::Interval, ::_IA_Oi, X::Integer) where T = Iterators.product(1:w.x-1, w.x+1:w.y-1)

## Enumerate accessible worlds from a set of worlds
enumAcc2(w::Interval, r::R where R<:_IARelation, X::Integer) where T = begin
	IterTools.imap(Interval,
		enumAccW2(w, r, X)
	)
end
enumAcc2(S::Union{WorldGenerator,AbstractWorldSet{Interval}}, r::R where R<:_IARelation, X::Integer) where T = begin
	# println("Fallback")
	IterTools.imap(Interval,
		IterTools.distinct(Iterators.flatten((enumAccW2(w, r, X) for w in S)))
	)
end


# More efficient implementations for edge cases
# This makes sense if we have 2-Tuples instead of intervals
# function snd((a,b)::Tuple) b end
# function fst((a,b)::Tuple) a end
# enumAcc2_1(S::Union{WorldGenerator,AbstractWorldSet{Interval}}, ::_IA_L, X::Integer) where T = 
# 	IterTools.imap(Interval,
# 		enumAccW2(S[argmin(map(snd, S))], IA_L, X)
# 	)
# enumAcc2_1(S::Union{WorldGenerator,AbstractWorldSet{Interval}}, ::_IA_Li, X::Integer) where T = 
# 	IterTools.imap(Interval,
# 		enumAccW2(S[argmax(map(fst, S))], IA_Li, X)
# 	)

# More efficient implementations for edge cases
enumAcc2_1_2(S::Union{WorldGenerator,AbstractWorldSet{Interval}}, ::_IA_L, X::Integer) where T = begin
	# @show Base.argmin((w.y for w in S))
	IterTools.imap(Interval,
		enumAccW2(Base.argmin((w.y for w in S)), IA_L, X)
	)
end
enumAcc2_1_2(S::Union{WorldGenerator,AbstractWorldSet{Interval}}, ::_IA_Li, X::Integer) where T = begin
	# @show Base.argmax((w.x for w in S))
	IterTools.imap(Interval,
		enumAccW2(Base.argmax((w.x for w in S)), IA_Li, X)
	)
end

# More efficient implementations for edge cases
enumAcc2_2(S::Union{WorldGenerator,AbstractWorldSet{Interval}}, ::_IA_L, X::Integer) where T = begin
	m = argmin(y.(S))
	IterTools.imap(Interval,
		enumAccW2([w for (i,w) in enumerate(S) if i == m][1], IA_L, X)
	)
	end
enumAcc2_2(S::Union{WorldGenerator,AbstractWorldSet{Interval}}, ::_IA_Li, X::Integer) where T = begin
	m = argmax(x.(S))
	IterTools.imap(Interval,
		enumAccW2([w for (i,w) in enumerate(S) if i == m][1], IA_Li, X)
	)
	end

# More efficient implementations for edge cases
enumAcc2_2_2(WorS::WorldOrSet{Interval}, ::_RelationAll, X::Integer) where T = begin
	IterTools.imap(Interval,
		enumIntervalsInRange(1, X+1)
	)
	end
enumAcc2_2_2(S::Union{WorldGenerator,AbstractWorldSet{Interval}}, ::_IA_L, X::Integer) where T = begin
	IterTools.imap(Interval,
		enumAccW2(nth(S, argmin(y.(S))), IA_L, X)
	)
	end
enumAcc2_2_2(S::Union{WorldGenerator,AbstractWorldSet{Interval}}, ::_IA_Li, X::Integer) where T = begin
	IterTools.imap(Interval,
		enumAccW2(nth(S, argmax(x.(S))), IA_Li, X)
	)
	end

#=

############################################
BEGIN Performance tuning
############################################

using Revise
using BenchmarkTools
include("DecisionTree.jl/src/ModalLogic.jl")


channel = fill(1, 40)
X = length(channel)
S = [Interval(15, 25)]
S1 = enumAcc1(S, IA_L, channel)
S2 = enumAcc2(S, IA_L, channel)
Sc = Array{Interval,1}(collect(S))

@btime enumAcc1(S1, IA_L,  X) |> collect;    			# 595.462 μs (7570 allocations: 281.19 KiB)
@btime enumAcc1(S2, IA_L,  X) |> collect;    			# 623.972 μs (8017 allocations: 418.33 KiB)
@btime enumAcc1_1(S1, IA_L,  X) |> collect;				# 230.507 μs (2174 allocations: 73.41 KiB)
@btime enumAcc1_1(S2, IA_L,  X) |> collect;				# 315.552 μs (3692 allocations: 281.48 KiB)
@btime enumAcc2(S1, IA_L,  X) |> collect;					# 315.185 μs (6931 allocations: 289.08 KiB)
@btime enumAcc2(S2, IA_L,  X) |> collect;					# 363.924 μs (7534 allocations: 695.56 KiB)
@btime enumAcc2_1_2(S1, IA_L,  X) |> collect; 		# 230.560 μs (2094 allocations: 70.91 KiB)
@btime enumAcc2_1_2(S2, IA_L,  X) |> collect; 		# 313.631 μs (3612 allocations: 278.98 KiB)
@btime enumAcc2_2(S1, IA_L,  X) |> collect;				# 190.924 μs (1691 allocations: 64.64 KiB)
@btime enumAcc2_2(S2, IA_L,  X) |> collect;				# 242.755 μs (2692 allocations: 193.08 KiB)
@btime enumAcc2_2_2(S1, IA_L,  X) |> collect;			# 77.094 μs (748 allocations: 31.86 KiB)
@btime enumAcc2_2_2(S2, IA_L,  X) |> collect;			# 103.703 μs (1199 allocations: 84.34 KiB)
#Array:
@btime enumAcc1(Sc, IA_L,  X) |> collect;					# 77.120 μs (656 allocations: 32.16 KiB)
@btime enumAcc1_1(Sc, IA_L,  X) |> collect;				# 7.658 μs (225 allocations: 9.13 KiB)
@btime enumAcc1_2(Sc, IA_L,  X) |> collect;				# 7.568 μs (226 allocations: 9.20 KiB)
@btime enumAcc2(Sc, IA_L,  X) |> collect;					# 100.595 μs (1228 allocations: 87.91 KiB)
@btime enumAcc2_1_2(Sc, IA_L,  X) |> collect;			# 2.640 μs (118 allocations: 5.78 KiB)
@btime enumAcc2_2(Sc, IA_L,  X) |> collect;				# 2.779 μs (126 allocations: 6.14 KiB)
@btime enumAcc2_2_2(Sc, IA_L,  X) |> collect;			# 2.270 μs (119 allocations: 5.86 KiB)

@btime enumAcc1(S1, IA_Li,  X) |> collect;				# 16.859 ms (237528 allocations: 7.83 MiB)
@btime enumAcc1(S2, IA_Li,  X) |> collect;				# 17.255 ms (237975 allocations: 10.58 MiB)
@btime enumAcc1_1(S1, IA_Li,  X) |> collect;			# 292.431 μs (3427 allocations: 126.66 KiB)
@btime enumAcc1_1(S2, IA_Li,  X) |> collect;			# 383.223 μs (4945 allocations: 334.73 KiB)
@btime enumAcc2(S1, IA_Li,  X) |> collect;				# 5.417 ms (207753 allocations: 7.60 MiB)
@btime enumAcc2(S2, IA_Li,  X) |> collect;				# 6.482 ms (209008 allocations: 17.50 MiB)
@btime enumAcc2_1_2(S1, IA_Li,  X) |> collect;		# 247.680 μs (2722 allocations: 104.63 KiB)
@btime enumAcc2_1_2(S2, IA_Li,  X) |> collect;		# 336.925 μs (4240 allocations: 312.70 KiB)
@btime enumAcc2_2(S1, IA_Li,  X) |> collect;			# 200.390 μs (2319 allocations: 98.36 KiB)
@btime enumAcc2_2(S2, IA_Li,  X) |> collect;			# 262.138 μs (3320 allocations: 226.80 KiB)
@btime enumAcc2_2_2(S1, IA_Li,  X) |> collect;		# 204.298 μs (2312 allocations: 98.08 KiB)
@btime enumAcc2_2_2(S2, IA_Li,  X) |> collect;		# 210.995 μs (2892 allocations: 191.97 KiB)
#Array:
@btime enumAcc1(Sc, IA_Li,  X) |> collect;				# 64.353 μs (572 allocations: 29.09 KiB)
@btime enumAcc1_1(Sc, IA_Li,  X) |> collect;			# 7.000 μs (197 allocations: 8.25 KiB)
@btime enumAcc1_2(Sc, IA_Li,  X) |> collect;			# 6.736 μs (198 allocations: 8.33 KiB)
@btime enumAcc2(Sc, IA_Li,  X) |> collect;				# 89.649 μs (1104 allocations: 78.56 KiB)
@btime enumAcc2_1_2(Sc, IA_Li,  X) |> collect;		# 2.313 μs (104 allocations: 5.34 KiB)
@btime enumAcc2_2(Sc, IA_Li,  X) |> collect;			# 2.588 μs (112 allocations: 5.70 KiB)
@btime enumAcc2_2_2(Sc, IA_Li,  X) |> collect;		# 2.097 μs (105 allocations: 5.42 KiB)

@btime enumAcc1(S1, IA_Di,  X) |> collect;				# 5.224 ms (67349 allocations: 2.27 MiB)
@btime enumAcc1(S2, IA_Di,  X) |> collect;				# 5.381 ms (67796 allocations: 3.10 MiB)
@btime enumAcc2(S1, IA_Di,  X) |> collect;				# 1.857 ms (60502 allocations: 2.26 MiB)
@btime enumAcc2(S2, IA_Di,  X) |> collect;				# 2.085 ms (61443 allocations: 5.27 MiB)
#Array:
@btime enumAcc1(Sc, IA_Di,  X) |> collect;				# 166.439 μs (1533 allocations: 78.50 KiB)
@btime enumAcc2(Sc, IA_Di,  X) |> collect;				# 210.711 μs (2778 allocations: 192.80 KiB)


Results (date 02/02/2020):

-> enumAcc1 and enumAcc2 are best for arrays and iterators, respectively
=#
enumAcc(S::AbstractWorldSet{Interval}, r::R where R<:_IARelation, X::Integer) where T = enumAcc1(S, r, X)
enumAcc(S::WorldGenerator,             r::R where R<:_IARelation, X::Integer) where T = enumAcc2(S, r, X)
enumAcc(S::Interval,                   r::R where R<:_IARelation, X::Integer) where T = enumAcc2(S, r, X)
#=
-> enumAcc1_1 is never better than enumAcc2_1
=#
#=
-> For iterators and arrays, enumAcc2_2_2 is probably the best IA_L/IA_Li enumerator
=#
enumAcc(WorS::WorldOrSet{Interval},    ::_RelationAll, X::Integer) where T = enumAcc2_2_2(WorS, RelationAll, X)
enumAcc(S::WorldGenerator,             ::_IA_L,        X::Integer) where T = enumAcc2_2_2(S, IA_L, X)
enumAcc(S::AbstractWorldSet{Interval}, ::_IA_L,        X::Integer) where T = enumAcc2_2_2(S, IA_L, X)
enumAcc(S::WorldGenerator,             ::_IA_Li,       X::Integer) where T = enumAcc2_2_2(S, IA_Li, X)
enumAcc(S::AbstractWorldSet{Interval}, ::_IA_Li,       X::Integer) where T = enumAcc2_2_2(S, IA_Li, X)

const IntervalOntology = Ontology(Interval,IARelations)


# TODO generalize this as NTuple{4,Interval} ?
# 2-dimensional Interval counterpart: combination of two Intervals 
struct Interval2D <: AbstractWorld
	x :: Interval
	y :: Interval
	Interval2D(x,y) = new(x,y)
	Interval2D(w::_emptyWorld) = new(Interval(w),Interval(w))
	Interval2D(w::_centeredWorld, X::Integer, Y::Integer) = new(Interval(w,X),Interval(w,Y))
end

Interval2D(params::Tuple{Interval,Interval}) = Interval2D(params...)
Interval2D(params::Tuple{Tuple{Integer,Integer},Tuple{Integer,Integer}}) = Interval2D(params...)

# map((x)->readWorld(Interval(x),[1,2,3,4,5]), enumIntervalsInRange(1,6) |> collect)
@inline readWorld(w::Interval2D, channel::MatricialChannel{T,2}) where {T} = @inbounds channel[w.y.x:w.y.y-1,w.x.x:w.x.y-1]

# Interval2D relation as the combination of two Interval relations
# struct _IA2DRelation <: AbstractRelation
	# x :: R1 where R1 <: AbstractRelation
	# y :: R2 where R2 <: AbstractRelation
# end
# TODO
# 1) try maybe saying that x and y are _IARelation. Warning, this is wrong because IA_I is not.
# 2) try dispatching like this:
struct _IA2DRelation{R1,R2} <: AbstractRelation
	x :: R1 #TODO? where R1<:AbstractRelation
	y :: R2 #TODO? where R2<:AbstractRelation
end

                                            const IA_IU  = _IA2DRelation(IA_I  , IA_U); const IA_IA  = _IA2DRelation(IA_I  , IA_A); const IA_IL  = _IA2DRelation(IA_I  , IA_L); const IA_IB  = _IA2DRelation(IA_I  , IA_B); const IA_IE  = _IA2DRelation(IA_I  , IA_E); const IA_ID  = _IA2DRelation(IA_I  , IA_D); const IA_IO  = _IA2DRelation(IA_I  , IA_O); const IA_IAi  = _IA2DRelation(IA_I  , IA_Ai); const IA_ILi  = _IA2DRelation(IA_I  , IA_Li); const IA_IBi  = _IA2DRelation(IA_I  , IA_Bi); const IA_IEi  = _IA2DRelation(IA_I  , IA_Ei); const IA_IDi  = _IA2DRelation(IA_I  , IA_Di); const IA_IOi  = _IA2DRelation(IA_I  , IA_Oi);
const IA_UI  = _IA2DRelation(IA_U  , IA_I); const IA_UU  = _IA2DRelation(IA_U  , IA_U); const IA_UA  = _IA2DRelation(IA_U  , IA_A); const IA_UL  = _IA2DRelation(IA_U  , IA_L); const IA_UB  = _IA2DRelation(IA_U  , IA_B); const IA_UE  = _IA2DRelation(IA_U  , IA_E); const IA_UD  = _IA2DRelation(IA_U  , IA_D); const IA_UO  = _IA2DRelation(IA_U  , IA_O); const IA_UAi  = _IA2DRelation(IA_U  , IA_Ai); const IA_ULi  = _IA2DRelation(IA_U  , IA_Li); const IA_UBi  = _IA2DRelation(IA_U  , IA_Bi); const IA_UEi  = _IA2DRelation(IA_U  , IA_Ei); const IA_UDi  = _IA2DRelation(IA_U  , IA_Di); const IA_UOi  = _IA2DRelation(IA_U  , IA_Oi);
const IA_AI  = _IA2DRelation(IA_A  , IA_I); const IA_AU  = _IA2DRelation(IA_A  , IA_U); const IA_AA  = _IA2DRelation(IA_A  , IA_A); const IA_AL  = _IA2DRelation(IA_A  , IA_L); const IA_AB  = _IA2DRelation(IA_A  , IA_B); const IA_AE  = _IA2DRelation(IA_A  , IA_E); const IA_AD  = _IA2DRelation(IA_A  , IA_D); const IA_AO  = _IA2DRelation(IA_A  , IA_O); const IA_AAi  = _IA2DRelation(IA_A  , IA_Ai); const IA_ALi  = _IA2DRelation(IA_A  , IA_Li); const IA_ABi  = _IA2DRelation(IA_A  , IA_Bi); const IA_AEi  = _IA2DRelation(IA_A  , IA_Ei); const IA_ADi  = _IA2DRelation(IA_A  , IA_Di); const IA_AOi  = _IA2DRelation(IA_A  , IA_Oi);
const IA_LI  = _IA2DRelation(IA_L  , IA_I); const IA_LU  = _IA2DRelation(IA_L  , IA_U); const IA_LA  = _IA2DRelation(IA_L  , IA_A); const IA_LL  = _IA2DRelation(IA_L  , IA_L); const IA_LB  = _IA2DRelation(IA_L  , IA_B); const IA_LE  = _IA2DRelation(IA_L  , IA_E); const IA_LD  = _IA2DRelation(IA_L  , IA_D); const IA_LO  = _IA2DRelation(IA_L  , IA_O); const IA_LAi  = _IA2DRelation(IA_L  , IA_Ai); const IA_LLi  = _IA2DRelation(IA_L  , IA_Li); const IA_LBi  = _IA2DRelation(IA_L  , IA_Bi); const IA_LEi  = _IA2DRelation(IA_L  , IA_Ei); const IA_LDi  = _IA2DRelation(IA_L  , IA_Di); const IA_LOi  = _IA2DRelation(IA_L  , IA_Oi);
const IA_BI  = _IA2DRelation(IA_B  , IA_I); const IA_BU  = _IA2DRelation(IA_B  , IA_U); const IA_BA  = _IA2DRelation(IA_B  , IA_A); const IA_BL  = _IA2DRelation(IA_B  , IA_L); const IA_BB  = _IA2DRelation(IA_B  , IA_B); const IA_BE  = _IA2DRelation(IA_B  , IA_E); const IA_BD  = _IA2DRelation(IA_B  , IA_D); const IA_BO  = _IA2DRelation(IA_B  , IA_O); const IA_BAi  = _IA2DRelation(IA_B  , IA_Ai); const IA_BLi  = _IA2DRelation(IA_B  , IA_Li); const IA_BBi  = _IA2DRelation(IA_B  , IA_Bi); const IA_BEi  = _IA2DRelation(IA_B  , IA_Ei); const IA_BDi  = _IA2DRelation(IA_B  , IA_Di); const IA_BOi  = _IA2DRelation(IA_B  , IA_Oi);
const IA_EI  = _IA2DRelation(IA_E  , IA_I); const IA_EU  = _IA2DRelation(IA_E  , IA_U); const IA_EA  = _IA2DRelation(IA_E  , IA_A); const IA_EL  = _IA2DRelation(IA_E  , IA_L); const IA_EB  = _IA2DRelation(IA_E  , IA_B); const IA_EE  = _IA2DRelation(IA_E  , IA_E); const IA_ED  = _IA2DRelation(IA_E  , IA_D); const IA_EO  = _IA2DRelation(IA_E  , IA_O); const IA_EAi  = _IA2DRelation(IA_E  , IA_Ai); const IA_ELi  = _IA2DRelation(IA_E  , IA_Li); const IA_EBi  = _IA2DRelation(IA_E  , IA_Bi); const IA_EEi  = _IA2DRelation(IA_E  , IA_Ei); const IA_EDi  = _IA2DRelation(IA_E  , IA_Di); const IA_EOi  = _IA2DRelation(IA_E  , IA_Oi);
const IA_DI  = _IA2DRelation(IA_D  , IA_I); const IA_DU  = _IA2DRelation(IA_D  , IA_U); const IA_DA  = _IA2DRelation(IA_D  , IA_A); const IA_DL  = _IA2DRelation(IA_D  , IA_L); const IA_DB  = _IA2DRelation(IA_D  , IA_B); const IA_DE  = _IA2DRelation(IA_D  , IA_E); const IA_DD  = _IA2DRelation(IA_D  , IA_D); const IA_DO  = _IA2DRelation(IA_D  , IA_O); const IA_DAi  = _IA2DRelation(IA_D  , IA_Ai); const IA_DLi  = _IA2DRelation(IA_D  , IA_Li); const IA_DBi  = _IA2DRelation(IA_D  , IA_Bi); const IA_DEi  = _IA2DRelation(IA_D  , IA_Ei); const IA_DDi  = _IA2DRelation(IA_D  , IA_Di); const IA_DOi  = _IA2DRelation(IA_D  , IA_Oi);
const IA_OI  = _IA2DRelation(IA_O  , IA_I); const IA_OU  = _IA2DRelation(IA_O  , IA_U); const IA_OA  = _IA2DRelation(IA_O  , IA_A); const IA_OL  = _IA2DRelation(IA_O  , IA_L); const IA_OB  = _IA2DRelation(IA_O  , IA_B); const IA_OE  = _IA2DRelation(IA_O  , IA_E); const IA_OD  = _IA2DRelation(IA_O  , IA_D); const IA_OO  = _IA2DRelation(IA_O  , IA_O); const IA_OAi  = _IA2DRelation(IA_O  , IA_Ai); const IA_OLi  = _IA2DRelation(IA_O  , IA_Li); const IA_OBi  = _IA2DRelation(IA_O  , IA_Bi); const IA_OEi  = _IA2DRelation(IA_O  , IA_Ei); const IA_ODi  = _IA2DRelation(IA_O  , IA_Di); const IA_OOi  = _IA2DRelation(IA_O  , IA_Oi);
const IA_AiI = _IA2DRelation(IA_Ai , IA_I); const IA_AiU = _IA2DRelation(IA_Ai , IA_U); const IA_AiA = _IA2DRelation(IA_Ai , IA_A); const IA_AiL = _IA2DRelation(IA_Ai , IA_L); const IA_AiB = _IA2DRelation(IA_Ai , IA_B); const IA_AiE = _IA2DRelation(IA_Ai , IA_E); const IA_AiD = _IA2DRelation(IA_Ai , IA_D); const IA_AiO = _IA2DRelation(IA_Ai , IA_O); const IA_AiAi = _IA2DRelation(IA_Ai , IA_Ai); const IA_AiLi = _IA2DRelation(IA_Ai , IA_Li); const IA_AiBi = _IA2DRelation(IA_Ai , IA_Bi); const IA_AiEi = _IA2DRelation(IA_Ai , IA_Ei); const IA_AiDi = _IA2DRelation(IA_Ai , IA_Di); const IA_AiOi = _IA2DRelation(IA_Ai , IA_Oi);
const IA_LiI = _IA2DRelation(IA_Li , IA_I); const IA_LiU = _IA2DRelation(IA_Li , IA_U); const IA_LiA = _IA2DRelation(IA_Li , IA_A); const IA_LiL = _IA2DRelation(IA_Li , IA_L); const IA_LiB = _IA2DRelation(IA_Li , IA_B); const IA_LiE = _IA2DRelation(IA_Li , IA_E); const IA_LiD = _IA2DRelation(IA_Li , IA_D); const IA_LiO = _IA2DRelation(IA_Li , IA_O); const IA_LiAi = _IA2DRelation(IA_Li , IA_Ai); const IA_LiLi = _IA2DRelation(IA_Li , IA_Li); const IA_LiBi = _IA2DRelation(IA_Li , IA_Bi); const IA_LiEi = _IA2DRelation(IA_Li , IA_Ei); const IA_LiDi = _IA2DRelation(IA_Li , IA_Di); const IA_LiOi = _IA2DRelation(IA_Li , IA_Oi);
const IA_BiI = _IA2DRelation(IA_Bi , IA_I); const IA_BiU = _IA2DRelation(IA_Bi , IA_U); const IA_BiA = _IA2DRelation(IA_Bi , IA_A); const IA_BiL = _IA2DRelation(IA_Bi , IA_L); const IA_BiB = _IA2DRelation(IA_Bi , IA_B); const IA_BiE = _IA2DRelation(IA_Bi , IA_E); const IA_BiD = _IA2DRelation(IA_Bi , IA_D); const IA_BiO = _IA2DRelation(IA_Bi , IA_O); const IA_BiAi = _IA2DRelation(IA_Bi , IA_Ai); const IA_BiLi = _IA2DRelation(IA_Bi , IA_Li); const IA_BiBi = _IA2DRelation(IA_Bi , IA_Bi); const IA_BiEi = _IA2DRelation(IA_Bi , IA_Ei); const IA_BiDi = _IA2DRelation(IA_Bi , IA_Di); const IA_BiOi = _IA2DRelation(IA_Bi , IA_Oi);
const IA_EiI = _IA2DRelation(IA_Ei , IA_I); const IA_EiU = _IA2DRelation(IA_Ei , IA_U); const IA_EiA = _IA2DRelation(IA_Ei , IA_A); const IA_EiL = _IA2DRelation(IA_Ei , IA_L); const IA_EiB = _IA2DRelation(IA_Ei , IA_B); const IA_EiE = _IA2DRelation(IA_Ei , IA_E); const IA_EiD = _IA2DRelation(IA_Ei , IA_D); const IA_EiO = _IA2DRelation(IA_Ei , IA_O); const IA_EiAi = _IA2DRelation(IA_Ei , IA_Ai); const IA_EiLi = _IA2DRelation(IA_Ei , IA_Li); const IA_EiBi = _IA2DRelation(IA_Ei , IA_Bi); const IA_EiEi = _IA2DRelation(IA_Ei , IA_Ei); const IA_EiDi = _IA2DRelation(IA_Ei , IA_Di); const IA_EiOi = _IA2DRelation(IA_Ei , IA_Oi);
const IA_DiI = _IA2DRelation(IA_Di , IA_I); const IA_DiU = _IA2DRelation(IA_Di , IA_U); const IA_DiA = _IA2DRelation(IA_Di , IA_A); const IA_DiL = _IA2DRelation(IA_Di , IA_L); const IA_DiB = _IA2DRelation(IA_Di , IA_B); const IA_DiE = _IA2DRelation(IA_Di , IA_E); const IA_DiD = _IA2DRelation(IA_Di , IA_D); const IA_DiO = _IA2DRelation(IA_Di , IA_O); const IA_DiAi = _IA2DRelation(IA_Di , IA_Ai); const IA_DiLi = _IA2DRelation(IA_Di , IA_Li); const IA_DiBi = _IA2DRelation(IA_Di , IA_Bi); const IA_DiEi = _IA2DRelation(IA_Di , IA_Ei); const IA_DiDi = _IA2DRelation(IA_Di , IA_Di); const IA_DiOi = _IA2DRelation(IA_Di , IA_Oi);
const IA_OiI = _IA2DRelation(IA_Oi , IA_I); const IA_OiU = _IA2DRelation(IA_Oi , IA_U); const IA_OiA = _IA2DRelation(IA_Oi , IA_A); const IA_OiL = _IA2DRelation(IA_Oi , IA_L); const IA_OiB = _IA2DRelation(IA_Oi , IA_B); const IA_OiE = _IA2DRelation(IA_Oi , IA_E); const IA_OiD = _IA2DRelation(IA_Oi , IA_D); const IA_OiO = _IA2DRelation(IA_Oi , IA_O); const IA_OiAi = _IA2DRelation(IA_Oi , IA_Ai); const IA_OiLi = _IA2DRelation(IA_Oi , IA_Li); const IA_OiBi = _IA2DRelation(IA_Oi , IA_Bi); const IA_OiEi = _IA2DRelation(IA_Oi , IA_Ei); const IA_OiDi = _IA2DRelation(IA_Oi , IA_Di); const IA_OiOi = _IA2DRelation(IA_Oi , IA_Oi);

print_rel_short(::_IA2DRelation{_XR,_YR}) where {_XR<:AbstractRelation,_YR<:AbstractRelation} =
	string(print_rel_short(_XR()), ",", print_rel_short(_YR())); 
#                                   print_rel_short(::_IA2DRelation{_IA_I ,_IA_U})  = "I,U";  print_rel_short(::_IA2DRelation{_IA_I ,_IA_A})  = "I,A";  print_rel_short(::_IA2DRelation{_IA_I ,_IA_L})  = "I,L";  print_rel_short(::_IA2DRelation{_IA_I ,_IA_B})  = "I,B";  print_rel_short(::_IA2DRelation{_IA_I ,_IA_E})  = "I,E";  print_rel_short(::_IA2DRelation{_IA_I ,_IA_D})  = "I,D";  print_rel_short(::_IA2DRelation{_IA_I ,_IA_O})  = "I,O";  print_rel_short(::_IA2DRelation{_IA_I ,_IA_Ai})  = "I,Ai";  print_rel_short(::_IA2DRelation{_IA_I ,_IA_Li})  = "I,Li";  print_rel_short(::_IA2DRelation{_IA_I ,_IA_Bi})  = "I,Bi";  print_rel_short(::_IA2DRelation{_IA_I ,_IA_Ei})  = "I,Ei";  print_rel_short(::_IA2DRelation{_IA_I ,_IA_Di})  = "I,Di";  print_rel_short(::_IA2DRelation{_IA_I ,_IA_Oi})  = "I,Oi";
# print_rel_short(::_IA2DRelation{_IA_U ,_IA_I})  = "U,I";  print_rel_short(::_IA2DRelation{_IA_U ,_IA_U})  = "U,U";  print_rel_short(::_IA2DRelation{_IA_U ,_IA_A})  = "U,A";  print_rel_short(::_IA2DRelation{_IA_U ,_IA_L})  = "U,L";  print_rel_short(::_IA2DRelation{_IA_U ,_IA_B})  = "U,B";  print_rel_short(::_IA2DRelation{_IA_U ,_IA_E})  = "U,E";  print_rel_short(::_IA2DRelation{_IA_U ,_IA_D})  = "U,D";  print_rel_short(::_IA2DRelation{_IA_U ,_IA_O})  = "U,O";  print_rel_short(::_IA2DRelation{_IA_U ,_IA_Ai})  = "U,Ai";  print_rel_short(::_IA2DRelation{_IA_U ,_IA_Li})  = "U,Li";  print_rel_short(::_IA2DRelation{_IA_U ,_IA_Bi})  = "U,Bi";  print_rel_short(::_IA2DRelation{_IA_U ,_IA_Ei})  = "U,Ei";  print_rel_short(::_IA2DRelation{_IA_U ,_IA_Di})  = "U,Di";  print_rel_short(::_IA2DRelation{_IA_U ,_IA_Oi})  = "U,Oi";
# print_rel_short(::_IA2DRelation{_IA_A ,_IA_I})  = "A,I";  print_rel_short(::_IA2DRelation{_IA_A ,_IA_U})  = "A,U";  print_rel_short(::_IA2DRelation{_IA_A ,_IA_A})  = "A,A";  print_rel_short(::_IA2DRelation{_IA_A ,_IA_L})  = "A,L";  print_rel_short(::_IA2DRelation{_IA_A ,_IA_B})  = "A,B";  print_rel_short(::_IA2DRelation{_IA_A ,_IA_E})  = "A,E";  print_rel_short(::_IA2DRelation{_IA_A ,_IA_D})  = "A,D";  print_rel_short(::_IA2DRelation{_IA_A ,_IA_O})  = "A,O";  print_rel_short(::_IA2DRelation{_IA_A ,_IA_Ai})  = "A,Ai";  print_rel_short(::_IA2DRelation{_IA_A ,_IA_Li})  = "A,Li";  print_rel_short(::_IA2DRelation{_IA_A ,_IA_Bi})  = "A,Bi";  print_rel_short(::_IA2DRelation{_IA_A ,_IA_Ei})  = "A,Ei";  print_rel_short(::_IA2DRelation{_IA_A ,_IA_Di})  = "A,Di";  print_rel_short(::_IA2DRelation{_IA_A ,_IA_Oi})  = "A,Oi";
# print_rel_short(::_IA2DRelation{_IA_L ,_IA_I})  = "L,I";  print_rel_short(::_IA2DRelation{_IA_L ,_IA_U})  = "L,U";  print_rel_short(::_IA2DRelation{_IA_L ,_IA_A})  = "L,A";  print_rel_short(::_IA2DRelation{_IA_L ,_IA_L})  = "L,L";  print_rel_short(::_IA2DRelation{_IA_L ,_IA_B})  = "L,B";  print_rel_short(::_IA2DRelation{_IA_L ,_IA_E})  = "L,E";  print_rel_short(::_IA2DRelation{_IA_L ,_IA_D})  = "L,D";  print_rel_short(::_IA2DRelation{_IA_L ,_IA_O})  = "L,O";  print_rel_short(::_IA2DRelation{_IA_L ,_IA_Ai})  = "L,Ai";  print_rel_short(::_IA2DRelation{_IA_L ,_IA_Li})  = "L,Li";  print_rel_short(::_IA2DRelation{_IA_L ,_IA_Bi})  = "L,Bi";  print_rel_short(::_IA2DRelation{_IA_L ,_IA_Ei})  = "L,Ei";  print_rel_short(::_IA2DRelation{_IA_L ,_IA_Di})  = "L,Di";  print_rel_short(::_IA2DRelation{_IA_L ,_IA_Oi})  = "L,Oi";
# print_rel_short(::_IA2DRelation{_IA_B ,_IA_I})  = "B,I";  print_rel_short(::_IA2DRelation{_IA_B ,_IA_U})  = "B,U";  print_rel_short(::_IA2DRelation{_IA_B ,_IA_A})  = "B,A";  print_rel_short(::_IA2DRelation{_IA_B ,_IA_L})  = "B,L";  print_rel_short(::_IA2DRelation{_IA_B ,_IA_B})  = "B,B";  print_rel_short(::_IA2DRelation{_IA_B ,_IA_E})  = "B,E";  print_rel_short(::_IA2DRelation{_IA_B ,_IA_D})  = "B,D";  print_rel_short(::_IA2DRelation{_IA_B ,_IA_O})  = "B,O";  print_rel_short(::_IA2DRelation{_IA_B ,_IA_Ai})  = "B,Ai";  print_rel_short(::_IA2DRelation{_IA_B ,_IA_Li})  = "B,Li";  print_rel_short(::_IA2DRelation{_IA_B ,_IA_Bi})  = "B,Bi";  print_rel_short(::_IA2DRelation{_IA_B ,_IA_Ei})  = "B,Ei";  print_rel_short(::_IA2DRelation{_IA_B ,_IA_Di})  = "B,Di";  print_rel_short(::_IA2DRelation{_IA_B ,_IA_Oi})  = "B,Oi";
# print_rel_short(::_IA2DRelation{_IA_E ,_IA_I})  = "E,I";  print_rel_short(::_IA2DRelation{_IA_E ,_IA_U})  = "E,U";  print_rel_short(::_IA2DRelation{_IA_E ,_IA_A})  = "E,A";  print_rel_short(::_IA2DRelation{_IA_E ,_IA_L})  = "E,L";  print_rel_short(::_IA2DRelation{_IA_E ,_IA_B})  = "E,B";  print_rel_short(::_IA2DRelation{_IA_E ,_IA_E})  = "E,E";  print_rel_short(::_IA2DRelation{_IA_E ,_IA_D})  = "E,D";  print_rel_short(::_IA2DRelation{_IA_E ,_IA_O})  = "E,O";  print_rel_short(::_IA2DRelation{_IA_E ,_IA_Ai})  = "E,Ai";  print_rel_short(::_IA2DRelation{_IA_E ,_IA_Li})  = "E,Li";  print_rel_short(::_IA2DRelation{_IA_E ,_IA_Bi})  = "E,Bi";  print_rel_short(::_IA2DRelation{_IA_E ,_IA_Ei})  = "E,Ei";  print_rel_short(::_IA2DRelation{_IA_E ,_IA_Di})  = "E,Di";  print_rel_short(::_IA2DRelation{_IA_E ,_IA_Oi})  = "E,Oi";
# print_rel_short(::_IA2DRelation{_IA_D ,_IA_I})  = "D,I";  print_rel_short(::_IA2DRelation{_IA_D ,_IA_U})  = "D,U";  print_rel_short(::_IA2DRelation{_IA_D ,_IA_A})  = "D,A";  print_rel_short(::_IA2DRelation{_IA_D ,_IA_L})  = "D,L";  print_rel_short(::_IA2DRelation{_IA_D ,_IA_B})  = "D,B";  print_rel_short(::_IA2DRelation{_IA_D ,_IA_E})  = "D,E";  print_rel_short(::_IA2DRelation{_IA_D ,_IA_D})  = "D,D";  print_rel_short(::_IA2DRelation{_IA_D ,_IA_O})  = "D,O";  print_rel_short(::_IA2DRelation{_IA_D ,_IA_Ai})  = "D,Ai";  print_rel_short(::_IA2DRelation{_IA_D ,_IA_Li})  = "D,Li";  print_rel_short(::_IA2DRelation{_IA_D ,_IA_Bi})  = "D,Bi";  print_rel_short(::_IA2DRelation{_IA_D ,_IA_Ei})  = "D,Ei";  print_rel_short(::_IA2DRelation{_IA_D ,_IA_Di})  = "D,Di";  print_rel_short(::_IA2DRelation{_IA_D ,_IA_Oi})  = "D,Oi";
# print_rel_short(::_IA2DRelation{_IA_O ,_IA_I})  = "O,I";  print_rel_short(::_IA2DRelation{_IA_O ,_IA_U})  = "O,U";  print_rel_short(::_IA2DRelation{_IA_O ,_IA_A})  = "O,A";  print_rel_short(::_IA2DRelation{_IA_O ,_IA_L})  = "O,L";  print_rel_short(::_IA2DRelation{_IA_O ,_IA_B})  = "O,B";  print_rel_short(::_IA2DRelation{_IA_O ,_IA_E})  = "O,E";  print_rel_short(::_IA2DRelation{_IA_O ,_IA_D})  = "O,D";  print_rel_short(::_IA2DRelation{_IA_O ,_IA_O})  = "O,O";  print_rel_short(::_IA2DRelation{_IA_O ,_IA_Ai})  = "O,Ai";  print_rel_short(::_IA2DRelation{_IA_O ,_IA_Li})  = "O,Li";  print_rel_short(::_IA2DRelation{_IA_O ,_IA_Bi})  = "O,Bi";  print_rel_short(::_IA2DRelation{_IA_O ,_IA_Ei})  = "O,Ei";  print_rel_short(::_IA2DRelation{_IA_O ,_IA_Di})  = "O,Di";  print_rel_short(::_IA2DRelation{_IA_O ,_IA_Oi})  = "O,Oi";
# print_rel_short(::_IA2DRelation{_IA_Ai,_IA_I}) = "Ai,I"; print_rel_short(::_IA2DRelation{_IA_Ai,_IA_U}) = "Ai,U"; print_rel_short(::_IA2DRelation{_IA_Ai,_IA_A}) = "Ai,A"; print_rel_short(::_IA2DRelation{_IA_Ai,_IA_L}) = "Ai,L"; print_rel_short(::_IA2DRelation{_IA_Ai,_IA_B}) = "Ai,B"; print_rel_short(::_IA2DRelation{_IA_Ai,_IA_E}) = "Ai,E"; print_rel_short(::_IA2DRelation{_IA_Ai,_IA_D}) = "Ai,D"; print_rel_short(::_IA2DRelation{_IA_Ai,_IA_O}) = "Ai,O"; print_rel_short(::_IA2DRelation{_IA_Ai,_IA_Ai}) = "Ai,Ai"; print_rel_short(::_IA2DRelation{_IA_Ai,_IA_Li}) = "Ai,Li"; print_rel_short(::_IA2DRelation{_IA_Ai,_IA_Bi}) = "Ai,Bi"; print_rel_short(::_IA2DRelation{_IA_Ai,_IA_Ei}) = "Ai,Ei"; print_rel_short(::_IA2DRelation{_IA_Ai,_IA_Di}) = "Ai,Di"; print_rel_short(::_IA2DRelation{_IA_Ai,_IA_Oi}) = "Ai,Oi";
# print_rel_short(::_IA2DRelation{_IA_Li,_IA_I}) = "Li,I"; print_rel_short(::_IA2DRelation{_IA_Li,_IA_U}) = "Li,U"; print_rel_short(::_IA2DRelation{_IA_Li,_IA_A}) = "Li,A"; print_rel_short(::_IA2DRelation{_IA_Li,_IA_L}) = "Li,L"; print_rel_short(::_IA2DRelation{_IA_Li,_IA_B}) = "Li,B"; print_rel_short(::_IA2DRelation{_IA_Li,_IA_E}) = "Li,E"; print_rel_short(::_IA2DRelation{_IA_Li,_IA_D}) = "Li,D"; print_rel_short(::_IA2DRelation{_IA_Li,_IA_O}) = "Li,O"; print_rel_short(::_IA2DRelation{_IA_Li,_IA_Ai}) = "Li,Ai"; print_rel_short(::_IA2DRelation{_IA_Li,_IA_Li}) = "Li,Li"; print_rel_short(::_IA2DRelation{_IA_Li,_IA_Bi}) = "Li,Bi"; print_rel_short(::_IA2DRelation{_IA_Li,_IA_Ei}) = "Li,Ei"; print_rel_short(::_IA2DRelation{_IA_Li,_IA_Di}) = "Li,Di"; print_rel_short(::_IA2DRelation{_IA_Li,_IA_Oi}) = "Li,Oi";
# print_rel_short(::_IA2DRelation{_IA_Bi,_IA_I}) = "Bi,I"; print_rel_short(::_IA2DRelation{_IA_Bi,_IA_U}) = "Bi,U"; print_rel_short(::_IA2DRelation{_IA_Bi,_IA_A}) = "Bi,A"; print_rel_short(::_IA2DRelation{_IA_Bi,_IA_L}) = "Bi,L"; print_rel_short(::_IA2DRelation{_IA_Bi,_IA_B}) = "Bi,B"; print_rel_short(::_IA2DRelation{_IA_Bi,_IA_E}) = "Bi,E"; print_rel_short(::_IA2DRelation{_IA_Bi,_IA_D}) = "Bi,D"; print_rel_short(::_IA2DRelation{_IA_Bi,_IA_O}) = "Bi,O"; print_rel_short(::_IA2DRelation{_IA_Bi,_IA_Ai}) = "Bi,Ai"; print_rel_short(::_IA2DRelation{_IA_Bi,_IA_Li}) = "Bi,Li"; print_rel_short(::_IA2DRelation{_IA_Bi,_IA_Bi}) = "Bi,Bi"; print_rel_short(::_IA2DRelation{_IA_Bi,_IA_Ei}) = "Bi,Ei"; print_rel_short(::_IA2DRelation{_IA_Bi,_IA_Di}) = "Bi,Di"; print_rel_short(::_IA2DRelation{_IA_Bi,_IA_Oi}) = "Bi,Oi";
# print_rel_short(::_IA2DRelation{_IA_Ei,_IA_I}) = "Ei,I"; print_rel_short(::_IA2DRelation{_IA_Ei,_IA_U}) = "Ei,U"; print_rel_short(::_IA2DRelation{_IA_Ei,_IA_A}) = "Ei,A"; print_rel_short(::_IA2DRelation{_IA_Ei,_IA_L}) = "Ei,L"; print_rel_short(::_IA2DRelation{_IA_Ei,_IA_B}) = "Ei,B"; print_rel_short(::_IA2DRelation{_IA_Ei,_IA_E}) = "Ei,E"; print_rel_short(::_IA2DRelation{_IA_Ei,_IA_D}) = "Ei,D"; print_rel_short(::_IA2DRelation{_IA_Ei,_IA_O}) = "Ei,O"; print_rel_short(::_IA2DRelation{_IA_Ei,_IA_Ai}) = "Ei,Ai"; print_rel_short(::_IA2DRelation{_IA_Ei,_IA_Li}) = "Ei,Li"; print_rel_short(::_IA2DRelation{_IA_Ei,_IA_Bi}) = "Ei,Bi"; print_rel_short(::_IA2DRelation{_IA_Ei,_IA_Ei}) = "Ei,Ei"; print_rel_short(::_IA2DRelation{_IA_Ei,_IA_Di}) = "Ei,Di"; print_rel_short(::_IA2DRelation{_IA_Ei,_IA_Oi}) = "Ei,Oi";
# print_rel_short(::_IA2DRelation{_IA_Di,_IA_I}) = "Di,I"; print_rel_short(::_IA2DRelation{_IA_Di,_IA_U}) = "Di,U"; print_rel_short(::_IA2DRelation{_IA_Di,_IA_A}) = "Di,A"; print_rel_short(::_IA2DRelation{_IA_Di,_IA_L}) = "Di,L"; print_rel_short(::_IA2DRelation{_IA_Di,_IA_B}) = "Di,B"; print_rel_short(::_IA2DRelation{_IA_Di,_IA_E}) = "Di,E"; print_rel_short(::_IA2DRelation{_IA_Di,_IA_D}) = "Di,D"; print_rel_short(::_IA2DRelation{_IA_Di,_IA_O}) = "Di,O"; print_rel_short(::_IA2DRelation{_IA_Di,_IA_Ai}) = "Di,Ai"; print_rel_short(::_IA2DRelation{_IA_Di,_IA_Li}) = "Di,Li"; print_rel_short(::_IA2DRelation{_IA_Di,_IA_Bi}) = "Di,Bi"; print_rel_short(::_IA2DRelation{_IA_Di,_IA_Ei}) = "Di,Ei"; print_rel_short(::_IA2DRelation{_IA_Di,_IA_Di}) = "Di,Di"; print_rel_short(::_IA2DRelation{_IA_Di,_IA_Oi}) = "Di,Oi";
# print_rel_short(::_IA2DRelation{_IA_Oi,_IA_I}) = "Oi,I"; print_rel_short(::_IA2DRelation{_IA_Oi,_IA_U}) = "Oi,U"; print_rel_short(::_IA2DRelation{_IA_Oi,_IA_A}) = "Oi,A"; print_rel_short(::_IA2DRelation{_IA_Oi,_IA_L}) = "Oi,L"; print_rel_short(::_IA2DRelation{_IA_Oi,_IA_B}) = "Oi,B"; print_rel_short(::_IA2DRelation{_IA_Oi,_IA_E}) = "Oi,E"; print_rel_short(::_IA2DRelation{_IA_Oi,_IA_D}) = "Oi,D"; print_rel_short(::_IA2DRelation{_IA_Oi,_IA_O}) = "Oi,O"; print_rel_short(::_IA2DRelation{_IA_Oi,_IA_Ai}) = "Oi,Ai"; print_rel_short(::_IA2DRelation{_IA_Oi,_IA_Li}) = "Oi,Li"; print_rel_short(::_IA2DRelation{_IA_Oi,_IA_Bi}) = "Oi,Bi"; print_rel_short(::_IA2DRelation{_IA_Oi,_IA_Ei}) = "Oi,Ei"; print_rel_short(::_IA2DRelation{_IA_Oi,_IA_Di}) = "Oi,Di"; print_rel_short(::_IA2DRelation{_IA_Oi,_IA_Oi}) = "Oi,Oi";

# Rectangle algebra
const IA2DRelations = [
       IA_IA ,IA_IL ,IA_IB ,IA_IE ,IA_ID ,IA_IO ,IA_IAi ,IA_ILi ,IA_IBi ,IA_IEi ,IA_IDi ,IA_IOi,
IA_AI ,IA_AA ,IA_AL ,IA_AB ,IA_AE ,IA_AD ,IA_AO ,IA_AAi ,IA_ALi ,IA_ABi ,IA_AEi ,IA_ADi ,IA_AOi,
IA_LI ,IA_LA ,IA_LL ,IA_LB ,IA_LE ,IA_LD ,IA_LO ,IA_LAi ,IA_LLi ,IA_LBi ,IA_LEi ,IA_LDi ,IA_LOi,
IA_BI ,IA_BA ,IA_BL ,IA_BB ,IA_BE ,IA_BD ,IA_BO ,IA_BAi ,IA_BLi ,IA_BBi ,IA_BEi ,IA_BDi ,IA_BOi,
IA_EI ,IA_EA ,IA_EL ,IA_EB ,IA_EE ,IA_ED ,IA_EO ,IA_EAi ,IA_ELi ,IA_EBi ,IA_EEi ,IA_EDi ,IA_EOi,
IA_DI ,IA_DA ,IA_DL ,IA_DB ,IA_DE ,IA_DD ,IA_DO ,IA_DAi ,IA_DLi ,IA_DBi ,IA_DEi ,IA_DDi ,IA_DOi,
IA_OI ,IA_OA ,IA_OL ,IA_OB ,IA_OE ,IA_OD ,IA_OO ,IA_OAi ,IA_OLi ,IA_OBi ,IA_OEi ,IA_ODi ,IA_OOi,
IA_AiI,IA_AiA,IA_AiL,IA_AiB,IA_AiE,IA_AiD,IA_AiO,IA_AiAi,IA_AiLi,IA_AiBi,IA_AiEi,IA_AiDi,IA_AiOi,
IA_LiI,IA_LiA,IA_LiL,IA_LiB,IA_LiE,IA_LiD,IA_LiO,IA_LiAi,IA_LiLi,IA_LiBi,IA_LiEi,IA_LiDi,IA_LiOi,
IA_BiI,IA_BiA,IA_BiL,IA_BiB,IA_BiE,IA_BiD,IA_BiO,IA_BiAi,IA_BiLi,IA_BiBi,IA_BiEi,IA_BiDi,IA_BiOi,
IA_EiI,IA_EiA,IA_EiL,IA_EiB,IA_EiE,IA_EiD,IA_EiO,IA_EiAi,IA_EiLi,IA_EiBi,IA_EiEi,IA_EiDi,IA_EiOi,
IA_DiI,IA_DiA,IA_DiL,IA_DiB,IA_DiE,IA_DiD,IA_DiO,IA_DiAi,IA_DiLi,IA_DiBi,IA_DiEi,IA_DiDi,IA_DiOi,
IA_OiI,IA_OiA,IA_OiL,IA_OiB,IA_OiE,IA_OiD,IA_OiO,IA_OiAi,IA_OiLi,IA_OiBi,IA_OiEi,IA_OiDi,IA_OiOi,
]

# Rectangle algebra coming TODO note that it also contains RelationAll
const IA2DRelations_extended = [
IA2DRelations..., IA_UU,
IA_UI ,IA_UA ,IA_UL ,IA_UB ,IA_UE ,IA_UD ,IA_UO ,IA_UAi ,IA_ULi ,IA_UBi ,IA_UEi ,IA_UDi ,IA_UOi,
IA_IU ,IA_AU ,IA_LU ,IA_BU ,IA_EU ,IA_DU ,IA_OU ,IA_AiU ,IA_LiU ,IA_BiU ,IA_EiU ,IA_DiU ,IA_OiU
]

# Interval Algebra 2D relations are computed by combining the two one-dimensional relations
# TODO check if dimensions are to be swapped
enumAcc(S::Any, r::R where R<:_IA2DRelation, X::Integer, Y::Integer) where T =
	IterTools.imap(Interval2D,
		Iterators.flatten(
			(Iterators.product(enumAcc(w.x, r.x, X), enumAcc(w.y, r.y, Y)) for w in S)
		)
)
enumAcc(WorS::WorldOrSet{Interval2D}, r::_RelationAll, X::Integer, Y::Integer) where T = enumAcc(WorS, IA_UU, X, Y)

const Interval2DOntology = Ontology(Interval2D,IA2DRelations)

worldTypeSize(::Type{Interval}) = 2
worldTypeSize(::Type{Interval2D}) = 4
getIntervalOntologyOfDim(::Val{1}) = IntervalOntology
getIntervalOntologyOfDim(::Val{2}) = Interval2DOntology
getIntervalOntologyOfDim(::MatricialDataset{T,D}) where {T,D} = getIntervalOntologyOfDim(Val(D-2))
