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
	Interval(::_InitialWorld) = new(-1,0)
end

Interval(params::Tuple{Integer,Integer}) = Interval(params...)
x(w::Interval) = w.x
y(w::Interval) = w.y

# map((x)->readWorld(Interval(x),[1,2,3,4,5]), enumIntervalsInRange(1,6) |> collect)
@inline readWorld(w::Interval, channel::MatricialChannel{T,1}) where {T} = channel[w.x:w.y-1]
@inline WMax(w::Interval, channel::MatricialChannel{T,1}) where {T} = maximum(readWorld(w,channel))
@inline WMin(w::Interval, channel::MatricialChannel{T,1}) where {T} = minimum(readWorld(w,channel))
@inline WLeq(w::Interval, channel::MatricialChannel{T,1}, val::Number) where T = begin # TODO maybe this becomes SIMD, or sum/all(readWorld(w,channel)  .<= val)
	# Source: https://stackoverflow.com/questions/47564825/check-if-all-the-elements-of-a-julia-array-are-equal
	@info "WLeq" w val readWorld(w,channel)
	@inbounds for x in readWorld(w,channel)
      x <= val || return false
  end
  return true
end


# 6+6 Interval relations
abstract type _IARelation <: AbstractRelation end
# TODO figure out what's the gain in using constant instances of these relations,
#  compared to using the type itself. Note: one should define the constant vector of instances IARelations here
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

const IARelations = [IA_A
											IA_L
											IA_B
											IA_E
											IA_D
											IA_O
											IA_Ai
											IA_Li
											IA_Bi
											IA_Ei
											IA_Di
											IA_Oi]

# Thought:
#  Domanda: ci serve iterare su generatori o no?
#  Nel learning filtro i mondi a seconda di quali soddisfano le clausole.
#  Posso farlo usando generatori, chissa', forse e' piu' conveniente?
#  Nel frattempo preparo il codice sia per generatori che per arrays

# Enumerate intervals in a given range
enumIntervalsInRange(a::Integer, b::Integer) =
	Iterators.filter((a)->a[1]<a[2], Iterators.product(a:b-1, a+1:b))

# enumAccW1(w::Interval, ::_RelationAll,    channel::MatricialChannel{T,1}) where T =
	# IterTools.imap(Interval, enumIntervalsInRange(1, length(channel)+1))

## Enumerate accessible worlds from a single world
enumAccW1(w::Interval, ::_IA_A,    channel::MatricialChannel{T,1}) where T =
	IterTools.imap((y)->Interval(w.y, y), w.y+1:length(channel)+1)
enumAccW1(w::Interval, ::_IA_Ai,   channel::MatricialChannel{T,1}) where T =
	IterTools.imap((x)->Interval(x, w.x), 1:w.x-1)
enumAccW1(w::Interval, ::_IA_L,    channel::MatricialChannel{T,1}) where T =
	IterTools.imap(Interval, enumIntervalsInRange(w.y+1, length(channel)+1))
enumAccW1(w::Interval, ::_IA_Li,   channel::MatricialChannel{T,1}) where T =
	IterTools.imap(Interval, enumIntervalsInRange(1, w.x-1))
enumAccW1(w::Interval, ::_IA_B,    channel::MatricialChannel{T,1}) where T =
	IterTools.imap((y)->Interval(w.x, y), w.x+1:w.y-1)
enumAccW1(w::Interval, ::_IA_Bi,   channel::MatricialChannel{T,1}) where T =
	IterTools.imap((y)->Interval(w.x, y), w.y+1:length(channel)+1)
enumAccW1(w::Interval, ::_IA_E,    channel::MatricialChannel{T,1}) where T =
	IterTools.imap((x)->Interval(x, w.y), w.x+1:w.y-1)
enumAccW1(w::Interval, ::_IA_Ei,   channel::MatricialChannel{T,1}) where T =
	IterTools.imap((x)->Interval(x, w.y), 1:w.x-1)
enumAccW1(w::Interval, ::_IA_D,    channel::MatricialChannel{T,1}) where T =
	IterTools.imap(Interval, enumIntervalsInRange(w.x+1, w.y-1))
enumAccW1(w::Interval, ::_IA_Di,   channel::MatricialChannel{T,1}) where T =
	IterTools.imap(Interval, Iterators.product(1:w.x-1, w.y+1:length(channel)+1))
enumAccW1(w::Interval, ::_IA_O,    channel::MatricialChannel{T,1}) where T =
	IterTools.imap(Interval, Iterators.product(w.x+1:w.y-1, w.y+1:length(channel)+1))
enumAccW1(w::Interval, ::_IA_Oi,   channel::MatricialChannel{T,1}) where T =
	IterTools.imap(Interval, Iterators.product(1:w.x-1, w.x+1:w.y-1))

## Enumerate accessible worlds from a set of worlds
enumAcc1(S::Union{WorldGenerator,WorldSet{Interval}}, r::R where R<:_IARelation, channel::MatricialChannel{T,1}) where T = begin
	IterTools.distinct(Iterators.flatten((enumAccW1(w, r, channel) for w in S)))
end

# More efficient implementations for edge cases

enumAcc1_1(S::Union{WorldGenerator,WorldSet{Interval}}, ::_IA_L, channel::MatricialChannel{T,1}) where T = begin
	# @show Base.argmin((w.y for w in S))
	enumAccW1(Base.argmin((w.y for w in S)), IA_L, channel)
end
enumAcc1_1(S::Union{WorldGenerator,WorldSet{Interval}}, ::_IA_Li, channel::MatricialChannel{T,1}) where T = begin
	# @show Base.argmax((w.x for w in S))
	enumAccW1(Base.argmax((w.x for w in S)), IA_Li, channel)
end
enumAcc1_2(S::Union{WorldGenerator,WorldSet{Interval}}, ::_IA_L, channel::MatricialChannel{T,1}) where T = 
	enumAccW1(S[argmin(y.(S))], IA_L, channel)
enumAcc1_2(S::Union{WorldGenerator,WorldSet{Interval}}, ::_IA_Li, channel::MatricialChannel{T,1}) where T = 
	enumAccW1(S[argmax(x.(S))], IA_Li, channel)

#####


# TODO check this other idea, maybe it's more efficient under certain conditions

# enumAccW2(w::Interval, ::_RelationAll,  channel::MatricialChannel{T,1}) where T = enumIntervalsInRange(1, length(channel)+1)
## Enumerate accessible worlds from a single world
enumAccW2(w::Interval, ::_IA_A,  channel::MatricialChannel{T,1}) where T = zip(Iterators.repeated(w.y), w.y+1:length(channel)+1)
enumAccW2(w::Interval, ::_IA_Ai, channel::MatricialChannel{T,1}) where T = zip(1:w.x-1, Iterators.repeated(w.x))
enumAccW2(w::Interval, ::_IA_L,  channel::MatricialChannel{T,1}) where T = enumIntervalsInRange(w.y+1, length(channel)+1)
enumAccW2(w::Interval, ::_IA_Li, channel::MatricialChannel{T,1}) where T = enumIntervalsInRange(1, w.x-1)
enumAccW2(w::Interval, ::_IA_B,  channel::MatricialChannel{T,1}) where T = zip(Iterators.repeated(w.x), w.x+1:w.y-1)
enumAccW2(w::Interval, ::_IA_Bi, channel::MatricialChannel{T,1}) where T = zip(Iterators.repeated(w.x), w.y+1:length(channel)+1)
enumAccW2(w::Interval, ::_IA_E,  channel::MatricialChannel{T,1}) where T = zip(w.x+1:w.y-1, Iterators.repeated(w.y))
enumAccW2(w::Interval, ::_IA_Ei, channel::MatricialChannel{T,1}) where T = zip(1:w.x-1, Iterators.repeated(w.y))
enumAccW2(w::Interval, ::_IA_D,  channel::MatricialChannel{T,1}) where T = enumIntervalsInRange(w.x+1, w.y-1)
enumAccW2(w::Interval, ::_IA_Di, channel::MatricialChannel{T,1}) where T = Iterators.product(1:w.x-1, w.y+1:length(channel)+1)
enumAccW2(w::Interval, ::_IA_O,  channel::MatricialChannel{T,1}) where T = Iterators.product(w.x+1:w.y-1, w.y+1:length(channel)+1)
enumAccW2(w::Interval, ::_IA_Oi, channel::MatricialChannel{T,1}) where T = Iterators.product(1:w.x-1, w.x+1:w.y-1)

## Enumerate accessible worlds from a set of worlds
enumAcc2(S::Union{WorldGenerator,WorldSet{Interval}}, r::R where R<:_IARelation, channel::MatricialChannel{T,1}) where T = begin
	# println("Fallback")
	IterTools.imap((params)->Interval(params...),
		IterTools.distinct(Iterators.flatten((enumAccW2(w, r, channel) for w in S))))
end


# More efficient implementations for edge cases
# This makes sense if we have 2-Tuples instead of intervals
# function snd((a,b)::Tuple) b end
# function fst((a,b)::Tuple) a end
# enumAcc2_1(S::Union{WorldGenerator,WorldSet{Interval}}, ::_IA_L, channel::MatricialChannel{T,1}) where T = 
# 	IterTools.imap((params)->Interval(params...),
# 		enumAccW2(S[argmin(map(snd, S))], IA_L, channel)
# 	)
# enumAcc2_1(S::Union{WorldGenerator,WorldSet{Interval}}, ::_IA_Li, channel::MatricialChannel{T,1}) where T = 
# 	IterTools.imap((params)->Interval(params...),
# 		enumAccW2(S[argmax(map(fst, S))], IA_Li, channel)
# 	)

# More efficient implementations for edge cases
enumAcc2_1_2(S::Union{WorldGenerator,WorldSet{Interval}}, ::_IA_L, channel::MatricialChannel{T,1}) where T = begin
	# @show Base.argmin((w.y for w in S))
	IterTools.imap((params)->Interval(params...),
		enumAccW2(Base.argmin((w.y for w in S)), IA_L, channel)
	)
end
enumAcc2_1_2(S::Union{WorldGenerator,WorldSet{Interval}}, ::_IA_Li, channel::MatricialChannel{T,1}) where T = begin
	# @show Base.argmax((w.x for w in S))
	IterTools.imap((params)->Interval(params...),
		enumAccW2(Base.argmax((w.x for w in S)), IA_Li, channel)
	)
end

# More efficient implementations for edge cases
enumAcc2_2(S::Union{WorldGenerator,WorldSet{Interval}}, ::_IA_L, channel::MatricialChannel{T,1}) where T = begin
	m = argmin(y.(S))
	IterTools.imap((params)->Interval(params...),
		enumAccW2([w for (i,w) in enumerate(S) if i == m][1], IA_L, channel)
	)
	end
enumAcc2_2(S::Union{WorldGenerator,WorldSet{Interval}}, ::_IA_Li, channel::MatricialChannel{T,1}) where T = begin
	m = argmax(x.(S))
	IterTools.imap((params)->Interval(params...),
		enumAccW2([w for (i,w) in enumerate(S) if i == m][1], IA_Li, channel)
	)
	end

# More efficient implementations for edge cases
enumAcc2_2_2(S::Union{WorldGenerator,WorldSet{Interval}}, ::_RelationAll, channel::MatricialChannel{T,1}) where T = begin
	IterTools.imap((params)->Interval(params...),
		enumIntervalsInRange(1, length(channel)+1)
	)
	end
enumAcc2_2_2(S::Union{WorldGenerator,WorldSet{Interval}}, ::_IA_L, channel::MatricialChannel{T,1}) where T = begin
	IterTools.imap((params)->Interval(params...),
		enumAccW2(nth(S, argmin(y.(S))), IA_L, channel)
	)
	end
enumAcc2_2_2(S::Union{WorldGenerator,WorldSet{Interval}}, ::_IA_Li, channel::MatricialChannel{T,1}) where T = begin
	IterTools.imap((params)->Interval(params...),
		enumAccW2(nth(S, argmax(x.(S))), IA_Li, channel)
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
S = [Interval(15, 25)]
S1 = enumAcc1(S, IA_L, channel)
S2 = enumAcc2(S, IA_L, channel)
Sc = Array{Interval,1}(collect(S))

@btime enumAcc1(S1, IA_L,  channel) |> collect;    			# 595.462 μs (7570 allocations: 281.19 KiB)
@btime enumAcc1(S2, IA_L,  channel) |> collect;    			# 623.972 μs (8017 allocations: 418.33 KiB)
@btime enumAcc1_1(S1, IA_L,  channel) |> collect;				# 230.507 μs (2174 allocations: 73.41 KiB)
@btime enumAcc1_1(S2, IA_L,  channel) |> collect;				# 315.552 μs (3692 allocations: 281.48 KiB)
@btime enumAcc2(S1, IA_L,  channel) |> collect;					# 315.185 μs (6931 allocations: 289.08 KiB)
@btime enumAcc2(S2, IA_L,  channel) |> collect;					# 363.924 μs (7534 allocations: 695.56 KiB)
@btime enumAcc2_1_2(S1, IA_L,  channel) |> collect; 		# 230.560 μs (2094 allocations: 70.91 KiB)
@btime enumAcc2_1_2(S2, IA_L,  channel) |> collect; 		# 313.631 μs (3612 allocations: 278.98 KiB)
@btime enumAcc2_2(S1, IA_L,  channel) |> collect;				# 190.924 μs (1691 allocations: 64.64 KiB)
@btime enumAcc2_2(S2, IA_L,  channel) |> collect;				# 242.755 μs (2692 allocations: 193.08 KiB)
@btime enumAcc2_2_2(S1, IA_L,  channel) |> collect;			# 77.094 μs (748 allocations: 31.86 KiB)
@btime enumAcc2_2_2(S2, IA_L,  channel) |> collect;			# 103.703 μs (1199 allocations: 84.34 KiB)
#Array:
@btime enumAcc1(Sc, IA_L,  channel) |> collect;					# 77.120 μs (656 allocations: 32.16 KiB)
@btime enumAcc1_1(Sc, IA_L,  channel) |> collect;				# 7.658 μs (225 allocations: 9.13 KiB)
@btime enumAcc1_2(Sc, IA_L,  channel) |> collect;				# 7.568 μs (226 allocations: 9.20 KiB)
@btime enumAcc2(Sc, IA_L,  channel) |> collect;					# 100.595 μs (1228 allocations: 87.91 KiB)
@btime enumAcc2_1_2(Sc, IA_L,  channel) |> collect;			# 2.640 μs (118 allocations: 5.78 KiB)
@btime enumAcc2_2(Sc, IA_L,  channel) |> collect;				# 2.779 μs (126 allocations: 6.14 KiB)
@btime enumAcc2_2_2(Sc, IA_L,  channel) |> collect;			# 2.270 μs (119 allocations: 5.86 KiB)

@btime enumAcc1(S1, IA_Li,  channel) |> collect;				# 16.859 ms (237528 allocations: 7.83 MiB)
@btime enumAcc1(S2, IA_Li,  channel) |> collect;				# 17.255 ms (237975 allocations: 10.58 MiB)
@btime enumAcc1_1(S1, IA_Li,  channel) |> collect;			# 292.431 μs (3427 allocations: 126.66 KiB)
@btime enumAcc1_1(S2, IA_Li,  channel) |> collect;			# 383.223 μs (4945 allocations: 334.73 KiB)
@btime enumAcc2(S1, IA_Li,  channel) |> collect;				# 5.417 ms (207753 allocations: 7.60 MiB)
@btime enumAcc2(S2, IA_Li,  channel) |> collect;				# 6.482 ms (209008 allocations: 17.50 MiB)
@btime enumAcc2_1_2(S1, IA_Li,  channel) |> collect;		# 247.680 μs (2722 allocations: 104.63 KiB)
@btime enumAcc2_1_2(S2, IA_Li,  channel) |> collect;		# 336.925 μs (4240 allocations: 312.70 KiB)
@btime enumAcc2_2(S1, IA_Li,  channel) |> collect;			# 200.390 μs (2319 allocations: 98.36 KiB)
@btime enumAcc2_2(S2, IA_Li,  channel) |> collect;			# 262.138 μs (3320 allocations: 226.80 KiB)
@btime enumAcc2_2_2(S1, IA_Li,  channel) |> collect;		# 204.298 μs (2312 allocations: 98.08 KiB)
@btime enumAcc2_2_2(S2, IA_Li,  channel) |> collect;		# 210.995 μs (2892 allocations: 191.97 KiB)
#Array:
@btime enumAcc1(Sc, IA_Li,  channel) |> collect;				# 64.353 μs (572 allocations: 29.09 KiB)
@btime enumAcc1_1(Sc, IA_Li,  channel) |> collect;			# 7.000 μs (197 allocations: 8.25 KiB)
@btime enumAcc1_2(Sc, IA_Li,  channel) |> collect;			# 6.736 μs (198 allocations: 8.33 KiB)
@btime enumAcc2(Sc, IA_Li,  channel) |> collect;				# 89.649 μs (1104 allocations: 78.56 KiB)
@btime enumAcc2_1_2(Sc, IA_Li,  channel) |> collect;		# 2.313 μs (104 allocations: 5.34 KiB)
@btime enumAcc2_2(Sc, IA_Li,  channel) |> collect;			# 2.588 μs (112 allocations: 5.70 KiB)
@btime enumAcc2_2_2(Sc, IA_Li,  channel) |> collect;		# 2.097 μs (105 allocations: 5.42 KiB)

@btime enumAcc1(S1, IA_Di,  channel) |> collect;				# 5.224 ms (67349 allocations: 2.27 MiB)
@btime enumAcc1(S2, IA_Di,  channel) |> collect;				# 5.381 ms (67796 allocations: 3.10 MiB)
@btime enumAcc2(S1, IA_Di,  channel) |> collect;				# 1.857 ms (60502 allocations: 2.26 MiB)
@btime enumAcc2(S2, IA_Di,  channel) |> collect;				# 2.085 ms (61443 allocations: 5.27 MiB)
#Array:
@btime enumAcc1(Sc, IA_Di,  channel) |> collect;				# 166.439 μs (1533 allocations: 78.50 KiB)
@btime enumAcc2(Sc, IA_Di,  channel) |> collect;				# 210.711 μs (2778 allocations: 192.80 KiB)


Results (date 02/02/2020):

-> enumAcc1 and enumAcc2 are best for arrays and iterators, respectively
=#
enumAcc(S::WorldSet{Interval}, r::R where R<:_IARelation, channel::MatricialChannel{T,1}) where T = enumAcc1(S, r, channel)
enumAcc(S::WorldGenerator, r::R where R<:_IARelation, channel::MatricialChannel{T,1}) where T = enumAcc2(S, r, channel)
#=
-> enumAcc1_1 is never better than enumAcc2_1
=#
#=
-> For iterators and arrays, enumAcc2_2_2 is probably the best IA_L/IA_Li enumerator
=#
enumAcc(S::WorldGenerator,     ::_RelationAll, channel::MatricialChannel{T,1}) where T = enumAcc2_2_2(S, RelationAll, channel)
enumAcc(S::WorldSet{Interval}, ::_RelationAll, channel::MatricialChannel{T,1}) where T = enumAcc2_2_2(S, RelationAll, channel)
enumAcc(S::WorldGenerator,     ::_IA_L, channel::MatricialChannel{T,1}) where T = enumAcc2_2_2(S, IA_L, channel)
enumAcc(S::WorldSet{Interval}, ::_IA_L, channel::MatricialChannel{T,1}) where T = enumAcc2_2_2(S, IA_L, channel)
enumAcc(S::WorldGenerator,     ::_IA_Li, channel::MatricialChannel{T,1}) where T = enumAcc2_2_2(S, IA_Li, channel)
enumAcc(S::WorldSet{Interval}, ::_IA_Li, channel::MatricialChannel{T,1}) where T = enumAcc2_2_2(S, IA_Li, channel)

const IntervalOntology = Ontology(Interval,IARelations)
