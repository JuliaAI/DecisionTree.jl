using IterTools
import Base.argmax
import Base.argmin
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
	world    :: Type{<:AbstractWorld}
	relation :: Type{<:AbstractRelation}
end

# An ontology interpreted over an N-dimensional domain gives rise to a Kripke model/frame.
# const MatricialDomain{T,N} = AbstractArray{T,N} end
struct OntologicalKripkeFrame{T,N}
	ontology  :: Ontology
	domain    :: AbstractArray{T,N}
end

# In the most generic case, a Kripke model/frame can be reprented in graph form.
# Thus, an "AbstractKripkeFrame" should also supertype some other representation.

# TODO Generalize World as a tuple of parameters ( https://stackoverflow.com/questions/40160120/generic-constructors-for-subtypes-of-an-abstract-type )

# Interval
struct Interval <: AbstractWorld
	x :: Integer
	y :: Integer
end

Interval(params::Tuple{Integer,Integer}) = Interval(params...)
x(w::Interval) = w.x
y(w::Interval) = w.y

# 6+6 Interval relations
abstract type IARelation <: AbstractRelation end
struct IA_A  <: IARelation end # After
struct IA_L  <: IARelation end # Later
struct IA_B  <: IARelation end # Begins
struct IA_E  <: IARelation end # Ends
struct IA_D  <: IARelation end # During
struct IA_O  <: IARelation end # Overlaps
struct IA_Ai <: IARelation end # inverse(After)
struct IA_Li <: IARelation end # inverse(Later)
struct IA_Bi <: IARelation end # inverse(Begins)
struct IA_Ei <: IARelation end # inverse(Ends)
struct IA_Di <: IARelation end # inverse(During)
struct IA_Oi <: IARelation end # inverse(Overlaps)

# Thought:
#  Domanda: ci serve iterare su generatori o no?
#  Nel learning filtro i mondi a seconda di quali soddisfano le clausole.
#  Posso farlo usando generatori, chissa', forse e' piu' conveniente?
#  Nel frattempo preparo il codice sia per generatori che per arrays

# Enumerate intervals in a given range
enumIntervalsInRange(a::Int, b::Int) =
	Iterators.filter((a)->a[1]<a[2], Iterators.product(a:b-1, a+1:b))

## Enumerate accessible worlds from a single world
enumAccW1(w::Interval, ::Type{IA_A},        N::Integer) =
	IterTools.imap((y)->Interval(w.y, y), w.y+1:N)
enumAccW1(w::Interval, ::Type{IA_Ai},        N::Integer) =
	IterTools.imap((x)->Interval(x, w.x), 1:w.x-1)
enumAccW1(w::Interval, ::Type{IA_L},        N::Integer) =
	IterTools.imap(Interval, enumIntervalsInRange(w.y+1, N))
enumAccW1(w::Interval, ::Type{IA_Li},        N::Integer) =
	IterTools.imap(Interval, enumIntervalsInRange(1, w.x-1))
enumAccW1(w::Interval, ::Type{IA_B},        N::Integer) =
	IterTools.imap((y)->Interval(w.x, y), w.x+1:w.y-1)
enumAccW1(w::Interval, ::Type{IA_Bi},        N::Integer) =
	IterTools.imap((y)->Interval(w.x, y), w.y+1:N)
enumAccW1(w::Interval, ::Type{IA_E},        N::Integer) =
	IterTools.imap((x)->Interval(x, w.y), w.x+1:w.y-1)
enumAccW1(w::Interval, ::Type{IA_Ei},        N::Integer) =
	IterTools.imap((x)->Interval(x, w.y), 1:w.x-1)
enumAccW1(w::Interval, ::Type{IA_D},        N::Integer) =
	IterTools.imap(Interval, enumIntervalsInRange(w.x+1, w.y-1))
enumAccW1(w::Interval, ::Type{IA_Di},        N::Integer) =
	IterTools.imap(Interval, Iterators.product(1:w.x-1, w.y+1:N))
enumAccW1(w::Interval, ::Type{IA_O},        N::Integer) =
	IterTools.imap(Interval, Iterators.product(w.x+1:w.y-1, w.y+1:N))
enumAccW1(w::Interval, ::Type{IA_Oi},        N::Integer) =
	IterTools.imap(Interval, Iterators.product(1:w.x-1, w.x+1:w.y-1))

## Enumerate accessible worlds from a set of worlds
enumAcc1(S::Union{Base.Generator,IterTools.Distinct,AbstractArray{Interval,1}}, r::Type{<:IARelation}, N::Integer) = begin
	IterTools.distinct(Iterators.flatten((enumAccW1(w, r, N) for w in S)))
end

# More efficient implementations for edge cases
enumAcc1_1(S::Union{Base.Generator,IterTools.Distinct,AbstractArray{Interval,1}}, ::Type{IA_L}, N::Integer) = begin
	# @show Base.argmin((w.y for w in S))
	enumAccW1(Base.argmin((w.y for w in S)), IA_L, N)
end
enumAcc1_1(S::Union{Base.Generator,IterTools.Distinct,AbstractArray{Interval,1}}, ::Type{IA_Li}, N::Integer) = begin
	# @show Base.argmax((w.x for w in S))
	enumAccW1(Base.argmax((w.x for w in S)), IA_Li, N)
end

# More efficient implementations for edge cases
enumAcc1_2(S::Union{Base.Generator,IterTools.Distinct,AbstractArray{Interval,1}}, ::Type{IA_L}, N::Integer) = 
	enumAccW1(S[argmin(y.(S))], IA_L, N)
enumAcc1_2(S::Union{Base.Generator,IterTools.Distinct,AbstractArray{Interval,1}}, ::Type{IA_Li}, N::Integer) = 
	enumAccW1(S[argmax(x.(S))], IA_Li, N)

const IntervalAlgebra = Ontology(Interval,IARelation)


#####


# TODO check this other idea, maybe it's more efficient under certain conditions

## Enumerate accessible worlds from a single world
enumAccW2(w::Interval, ::Type{IA_A},  N::Integer) = zip(Iterators.repeated(w.y), w.y+1:N)
enumAccW2(w::Interval, ::Type{IA_Ai}, N::Integer) = zip(1:w.x-1, Iterators.repeated(w.x))
enumAccW2(w::Interval, ::Type{IA_L},  N::Integer) = enumIntervalsInRange(w.y+1, N)
enumAccW2(w::Interval, ::Type{IA_Li}, N::Integer) = enumIntervalsInRange(1, w.x-1)
enumAccW2(w::Interval, ::Type{IA_B},  N::Integer) = zip(Iterators.repeated(w.x), w.x+1:w.y-1)
enumAccW2(w::Interval, ::Type{IA_Bi}, N::Integer) = zip(Iterators.repeated(w.x), w.y+1:N)
enumAccW2(w::Interval, ::Type{IA_E},  N::Integer) = zip(w.x+1:w.y-1, Iterators.repeated(w.y))
enumAccW2(w::Interval, ::Type{IA_Ei}, N::Integer) = zip(1:w.x-1, Iterators.repeated(w.y))
enumAccW2(w::Interval, ::Type{IA_D},  N::Integer) = enumIntervalsInRange(w.x+1, w.y-1)
enumAccW2(w::Interval, ::Type{IA_Di}, N::Integer) = Iterators.product(1:w.x-1, w.y+1:N)
enumAccW2(w::Interval, ::Type{IA_O},  N::Integer) = Iterators.product(w.x+1:w.y-1, w.y+1:N)
enumAccW2(w::Interval, ::Type{IA_Oi}, N::Integer) = Iterators.product(1:w.x-1, w.x+1:w.y-1)

## Enumerate accessible worlds from a set of worlds
enumAcc2(S::Union{Base.Generator,IterTools.Distinct,AbstractArray{Interval,1}}, r::Type{<:IARelation}, N::Integer) = begin
	# println("Fallback")
	IterTools.imap((params)->Interval(params...),
		IterTools.distinct(Iterators.flatten((enumAccW2(w, r, N) for w in S))))
end


# More efficient implementations for edge cases
# This makes sense if we have 2-Tuples instead of intervals
# function snd((a,b)::Tuple) b end
# function fst((a,b)::Tuple) a end
# enumAcc2_1(S::Union{Base.Generator,IterTools.Distinct,AbstractArray{Interval,1}}, ::Type{IA_L}, N::Integer) = 
# 	IterTools.imap((params)->Interval(params...),
# 		enumAccW2(S[argmin(map(snd, S))], IA_L, N)
# 	)
# enumAcc2_1(S::Union{Base.Generator,IterTools.Distinct,AbstractArray{Interval,1}}, ::Type{IA_Li}, N::Integer) = 
# 	IterTools.imap((params)->Interval(params...),
# 		enumAccW2(S[argmax(map(fst, S))], IA_Li, N)
# 	)

# More efficient implementations for edge cases
enumAcc2_1_2(S::Union{Base.Generator,IterTools.Distinct,AbstractArray{Interval,1}}, ::Type{IA_L}, N::Integer) = begin
	# @show Base.argmin((w.y for w in S))
	IterTools.imap((params)->Interval(params...),
		enumAccW2(Base.argmin((w.y for w in S)), IA_L, N)
	)
end
enumAcc2_1_2(S::Union{Base.Generator,IterTools.Distinct,AbstractArray{Interval,1}}, ::Type{IA_Li}, N::Integer) = begin
	# @show Base.argmax((w.x for w in S))
	IterTools.imap((params)->Interval(params...),
		enumAccW2(Base.argmax((w.x for w in S)), IA_Li, N)
	)
end

# More efficient implementations for edge cases
enumAcc2_2(S::Union{Base.Generator,IterTools.Distinct,AbstractArray{Interval,1}}, ::Type{IA_L}, N::Integer) = begin
	m = argmin(y.(S))
	IterTools.imap((params)->Interval(params...),
		enumAccW2([w for (i,w) in enumerate(S) if i == m][1], IA_L, N)
	)
	end
enumAcc2_2(S::Union{Base.Generator,IterTools.Distinct,AbstractArray{Interval,1}}, ::Type{IA_Li}, N::Integer) = begin
	m = argmax(x.(S))
	IterTools.imap((params)->Interval(params...),
		enumAccW2([w for (i,w) in enumerate(S) if i == m][1], IA_Li, N)
	)
	end

# More efficient implementations for edge cases
enumAcc2_2_2(S::Union{Base.Generator,IterTools.Distinct,AbstractArray{Interval,1}}, ::Type{IA_L}, N::Integer) = begin
	IterTools.imap((params)->Interval(params...),
		enumAccW2(nth(S, argmin(y.(S))), IA_L, N)
	)
	end
enumAcc2_2_2(S::Union{Base.Generator,IterTools.Distinct,AbstractArray{Interval,1}}, ::Type{IA_Li}, N::Integer) = begin
	IterTools.imap((params)->Interval(params...),
		enumAccW2(nth(S, argmax(x.(S))), IA_Li, N)
	)
	end

#=

############################################
BEGIN Performance tuning
############################################

using Revise
using BenchmarkTools
include("DecisionTree.jl/src/ModalLogic.jl")


N = 40
S = [Interval(15, 25)]
S1 = enumAcc1(S, IA_L, N)
S2 = enumAcc2(S, IA_L, N)
Sc = Array{Interval,1}(collect(S))

@btime enumAcc1(S1, IA_L,  N) |> collect;    			# 595.462 μs (7570 allocations: 281.19 KiB)
@btime enumAcc1(S2, IA_L,  N) |> collect;    			# 623.972 μs (8017 allocations: 418.33 KiB)
@btime enumAcc1_1(S1, IA_L,  N) |> collect;				# 230.507 μs (2174 allocations: 73.41 KiB)
@btime enumAcc1_1(S2, IA_L,  N) |> collect;				# 315.552 μs (3692 allocations: 281.48 KiB)
@btime enumAcc2(S1, IA_L,  N) |> collect;					# 315.185 μs (6931 allocations: 289.08 KiB)
@btime enumAcc2(S2, IA_L,  N) |> collect;					# 363.924 μs (7534 allocations: 695.56 KiB)
@btime enumAcc2_1_2(S1, IA_L,  N) |> collect; 		# 230.560 μs (2094 allocations: 70.91 KiB)
@btime enumAcc2_1_2(S2, IA_L,  N) |> collect; 		# 313.631 μs (3612 allocations: 278.98 KiB)
@btime enumAcc2_2(S1, IA_L,  N) |> collect;				# 190.924 μs (1691 allocations: 64.64 KiB)
@btime enumAcc2_2(S2, IA_L,  N) |> collect;				# 242.755 μs (2692 allocations: 193.08 KiB)
@btime enumAcc2_2_2(S1, IA_L,  N) |> collect;			# 77.094 μs (748 allocations: 31.86 KiB)
@btime enumAcc2_2_2(S2, IA_L,  N) |> collect;			# 103.703 μs (1199 allocations: 84.34 KiB)
#Array:
@btime enumAcc1(Sc, IA_L,  N) |> collect;					# 77.120 μs (656 allocations: 32.16 KiB)
@btime enumAcc1_1(Sc, IA_L,  N) |> collect;				# 7.658 μs (225 allocations: 9.13 KiB)
@btime enumAcc1_2(Sc, IA_L,  N) |> collect;				# 7.568 μs (226 allocations: 9.20 KiB)
@btime enumAcc2(Sc, IA_L,  N) |> collect;					# 100.595 μs (1228 allocations: 87.91 KiB)
@btime enumAcc2_1_2(Sc, IA_L,  N) |> collect;			# 2.640 μs (118 allocations: 5.78 KiB)
@btime enumAcc2_2(Sc, IA_L,  N) |> collect;				# 2.779 μs (126 allocations: 6.14 KiB)
@btime enumAcc2_2_2(Sc, IA_L,  N) |> collect;			# 2.270 μs (119 allocations: 5.86 KiB)

@btime enumAcc1(S1, IA_Li,  N) |> collect;				# 16.859 ms (237528 allocations: 7.83 MiB)
@btime enumAcc1(S2, IA_Li,  N) |> collect;				# 17.255 ms (237975 allocations: 10.58 MiB)
@btime enumAcc1_1(S1, IA_Li,  N) |> collect;			# 292.431 μs (3427 allocations: 126.66 KiB)
@btime enumAcc1_1(S2, IA_Li,  N) |> collect;			# 383.223 μs (4945 allocations: 334.73 KiB)
@btime enumAcc2(S1, IA_Li,  N) |> collect;				# 5.417 ms (207753 allocations: 7.60 MiB)
@btime enumAcc2(S2, IA_Li,  N) |> collect;				# 6.482 ms (209008 allocations: 17.50 MiB)
@btime enumAcc2_1_2(S1, IA_Li,  N) |> collect;		# 247.680 μs (2722 allocations: 104.63 KiB)
@btime enumAcc2_1_2(S2, IA_Li,  N) |> collect;		# 336.925 μs (4240 allocations: 312.70 KiB)
@btime enumAcc2_2(S1, IA_Li,  N) |> collect;			# 200.390 μs (2319 allocations: 98.36 KiB)
@btime enumAcc2_2(S2, IA_Li,  N) |> collect;			# 262.138 μs (3320 allocations: 226.80 KiB)
@btime enumAcc2_2_2(S1, IA_Li,  N) |> collect;		# 204.298 μs (2312 allocations: 98.08 KiB)
@btime enumAcc2_2_2(S2, IA_Li,  N) |> collect;		# 210.995 μs (2892 allocations: 191.97 KiB)
#Array:
@btime enumAcc1(Sc, IA_Li,  N) |> collect;				# 64.353 μs (572 allocations: 29.09 KiB)
@btime enumAcc1_1(Sc, IA_Li,  N) |> collect;			# 7.000 μs (197 allocations: 8.25 KiB)
@btime enumAcc1_2(Sc, IA_Li,  N) |> collect;			# 6.736 μs (198 allocations: 8.33 KiB)
@btime enumAcc2(Sc, IA_Li,  N) |> collect;				# 89.649 μs (1104 allocations: 78.56 KiB)
@btime enumAcc2_1_2(Sc, IA_Li,  N) |> collect;		# 2.313 μs (104 allocations: 5.34 KiB)
@btime enumAcc2_2(Sc, IA_Li,  N) |> collect;			# 2.588 μs (112 allocations: 5.70 KiB)
@btime enumAcc2_2_2(Sc, IA_Li,  N) |> collect;		# 2.097 μs (105 allocations: 5.42 KiB)

@btime enumAcc1(S1, IA_Di,  N) |> collect;				# 5.224 ms (67349 allocations: 2.27 MiB)
@btime enumAcc1(S2, IA_Di,  N) |> collect;				# 5.381 ms (67796 allocations: 3.10 MiB)
@btime enumAcc2(S1, IA_Di,  N) |> collect;				# 1.857 ms (60502 allocations: 2.26 MiB)
@btime enumAcc2(S2, IA_Di,  N) |> collect;				# 2.085 ms (61443 allocations: 5.27 MiB)
#Array:
@btime enumAcc1(Sc, IA_Di,  N) |> collect;				# 166.439 μs (1533 allocations: 78.50 KiB)
@btime enumAcc2(Sc, IA_Di,  N) |> collect;				# 210.711 μs (2778 allocations: 192.80 KiB)


Results (date 02/02/2020):

-> enumAcc1 and enumAcc2 are best for arrays and iterators, respectively
=#
enumAcc(S::AbstractArray{Interval,1}, r::Type{<:IARelation}, N::Integer) = enumAcc1(S, r, N)
enumAcc(S::Union{Base.Generator,IterTools.Distinct}, r::Type{<:IARelation}, N::Integer) = enumAcc2(S, r, N)
#=
-> enumAcc1_1 is never better than enumAcc2_1
=#
#=
-> For iterators and arrays, enumAcc2_2_2 is probably the best IA_L/IA_Li enumerator
=#
enumAcc(S::Union{Base.Generator,IterTools.Distinct,AbstractArray{Interval,1}}, r::Type{IA_Li}, N::Integer) = enumAcc2_2_2(S, Type{IA_Li}, N)
#=
############################################
END Performance tuning
############################################
=#

#=
TODO next
# 2D Interval counterpart Rectangle parallel
# struct ParRectangle <: AbstractWorld
# 	h :: Interval
# 	v :: Interval
# end

# const IntervalAlgebra = AbstractOntology(ParRectangle,RARelation)

=#
