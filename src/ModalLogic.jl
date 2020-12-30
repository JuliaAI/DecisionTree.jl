using IterTools

# Abstract class for Kripke frames
# abstract type AbstractKripkeFrame{T} end

# # Generic Kripke frame: worlds & relations
# struct KripkeFrame{T} <: AbstractKripkeFrame{T}
# 	# Majority class/value (output)
# 	worlds :: AbstractVector{T}
# 	# Training support
# 	values :: Vector{T}
# end

abstract type World end
abstract type Relation end

# Interval
struct Interval <: World
	x :: Integer
	y :: Integer
end

Interval(params::Tuple{Integer,Integer}) = Interval(params...)
x(w::Interval) = w.x
y(w::Interval) = w.y

# 6+6 Interval relations
abstract type IARelation <: Relation end
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

# Enumerate intervals in a given range
enumIntervalsInRange(a::Int, b::Int) =
	Iterators.filter((a)->a[1]<a[2], Iterators.product(a:b-1, a+1:b))

## Enumerate accessible worlds from a single world
enumAcc(w::Interval, ::Type{IA_A},        N::Integer) =
	IterTools.imap((y)->Interval(w.y, y), w.y+1:N)
enumAcc(w::Interval, ::Type{IA_Ai},        N::Integer) =
	IterTools.imap((x)->Interval(x, w.x), 1:w.x-1)
enumAcc(w::Interval, ::Type{IA_L},        N::Integer) =
	IterTools.imap(Interval, enumIntervalsInRange(w.y+1, N))
enumAcc(w::Interval, ::Type{IA_Li},        N::Integer) =
	IterTools.imap(Interval, enumIntervalsInRange(1, w.x-1))
enumAcc(w::Interval, ::Type{IA_B},        N::Integer) =
	IterTools.imap((y)->Interval(w.x, y), w.x+1:w.y-1)
enumAcc(w::Interval, ::Type{IA_Bi},        N::Integer) =
	IterTools.imap((y)->Interval(w.x, y), w.y+1:N)
enumAcc(w::Interval, ::Type{IA_E},        N::Integer) =
	IterTools.imap((x)->Interval(x, w.y), w.x+1:w.y-1)
enumAcc(w::Interval, ::Type{IA_Ei},        N::Integer) =
	IterTools.imap((x)->Interval(x, w.y), 1:w.x-1)
enumAcc(w::Interval, ::Type{IA_D},        N::Integer) =
	IterTools.imap(Interval, enumIntervalsInRange(w.x+1, w.y-1))
enumAcc(w::Interval, ::Type{IA_Di},        N::Integer) =
	IterTools.imap(Interval, Iterators.product(1:w.x-1, w.y+1:N))
enumAcc(w::Interval, ::Type{IA_O},        N::Integer) =
	IterTools.imap(Interval, Iterators.product(w.x+1:w.y-1, w.y+1:N))
enumAcc(w::Interval, ::Type{IA_Oi},        N::Integer) =
	IterTools.imap(Interval, Iterators.product(1:w.x-1, w.x+1:w.y-1))

## Enumerate accessible worlds from a set of worlds
enumAcc(S::Union{Base.Generator,AbstractArray{Interval}}, r::Type{<:IARelation}, N::Integer) = begin
	IterTools.distinct(Iterators.flatten((enumAcc(w, r, N) for w in S)))
end

# More efficient implementations for edge cases
enumAcc(S::AbstractArray{Interval}, ::Type{IA_L}, N::Integer) = 
	enumAcc(S[argmin(y.(S))], IA_L, N)
enumAcc(S::AbstractArray{Interval}, ::Type{IA_Li}, N::Integer) = 
	enumAcc(S[argmax(x.(S))], IA_Li, N)


# 2D Interval counterpart Rectangle parallel
struct ParallelRectangle <: World
	h :: Interval
	v :: Interval
end



#=

#####


# TODO check this other idea, maybe it's more efficient under certain conditions

## Enumerate accessible worlds from a single world
enumAccW(w::Interval, ::Type{IA_A}, N::Integer) = zip(Iterators.repeated(w.y), w.y+1:N)
enumAccW(w::Interval, ::Type{IA_Ai},N::Integer) = zip(1:w.x-1, Iterators.repeated(w.x))
enumAccW(w::Interval, ::Type{IA_L}, N::Integer) = enumIntervalsInRange(w.y+1, N)
enumAccW(w::Interval, ::Type{IA_Li},N::Integer) = enumIntervalsInRange(1, w.x-1)
enumAccW(w::Interval, ::Type{IA_B}, N::Integer) = zip(Iterators.repeated(w.x), w.x+1:w.y-1)
enumAccW(w::Interval, ::Type{IA_Bi},N::Integer) = zip(Iterators.repeated(w.x), w.y+1:N)
enumAccW(w::Interval, ::Type{IA_E}, N::Integer) = zip(w.x+1:w.y-1, Iterators.repeated(w.y))
enumAccW(w::Interval, ::Type{IA_Ei},N::Integer) = zip(1:w.x-1, Iterators.repeated(w.y))
enumAccW(w::Interval, ::Type{IA_D}, N::Integer) = enumIntervalsInRange(w.x+1, w.y-1)
enumAccW(w::Interval, ::Type{IA_Di},N::Integer) = Iterators.product(1:w.x-1, w.y+1:N)
enumAccW(w::Interval, ::Type{IA_O}, N::Integer) = Iterators.product(w.x+1:w.y-1, w.y+1:N)
enumAccW(w::Interval, ::Type{IA_Oi},N::Integer) = Iterators.product(1:w.x-1, w.x+1:w.y-1)

## Enumerate accessible worlds from a set of worlds
enumAcc2(S::Union{Base.Generator,AbstractArray{Interval}}, r::Type{<:IARelation}, N::Integer) = begin
	# println("Fallback")
	IterTools.imap(Interval,
		IterTools.distinct(Iterators.flatten((enumAccW(w, r, N) for w in S))))
end

# More efficient implementations for edge cases
# TODO these only work for arrays. Maybe I also need this for generators
enumAcc2(S::AbstractArray{Interval}, ::Type{IA_L}, N::Integer) = 
	IterTools.imap(Interval,
		enumAccW(S[argmin(y.(S))], IA_L, N))
enumAcc2(S::AbstractArray{Interval}, ::Type{IA_Li}, N::Integer) = 
	IterTools.imap(Interval,
		enumAccW(S[argmax(x.(S))], IA_Li, N))

#####

N = 40
S = [Interval(15, 25)]
S = enumAcc(S, IA_L, N)
Sc = collect(S)

@btime enumAcc(S, IA_A,  N) |> collect;
@btime enumAcc2(S, IA_A,  N) |> collect;
@btime enumAcc(S, IA_Ai, N) |> collect;
@btime enumAcc2(S, IA_Ai, N) |> collect;
@btime enumAcc(S, IA_L,  N) |> collect;
@btime enumAcc2(S, IA_L,  N) |> collect;
@btime enumAcc(S, IA_Li, N) |> collect;
@btime enumAcc2(S, IA_Li, N) |> collect;
@btime enumAcc(S, IA_B,  N) |> collect;
@btime enumAcc2(S, IA_B,  N) |> collect;
@btime enumAcc(S, IA_Bi, N) |> collect;
@btime enumAcc2(S, IA_Bi, N) |> collect;
@btime enumAcc(S, IA_E,  N) |> collect;
@btime enumAcc2(S, IA_E,  N) |> collect;
@btime enumAcc(S, IA_Ei, N) |> collect;
@btime enumAcc2(S, IA_Ei, N) |> collect;
@btime enumAcc(S, IA_D,  N) |> collect;
@btime enumAcc2(S, IA_D,  N) |> collect;
@btime enumAcc(S, IA_Di, N) |> collect;
@btime enumAcc2(S, IA_Di, N) |> collect;
@btime enumAcc(S, IA_O,  N) |> collect;
@btime enumAcc2(S, IA_O,  N) |> collect;
@btime enumAcc(S, IA_Oi, N) |> collect;
@btime enumAcc2(S, IA_Oi, N) |> collect;

@btime enumAcc(Sc, IA_A,  N) |> collect;
@btime enumAcc2(Sc, IA_A,  N) |> collect;
@btime enumAcc(Sc, IA_Ai, N) |> collect;
@btime enumAcc2(Sc, IA_Ai, N) |> collect;
@btime enumAcc(Sc, IA_L,  N) |> collect;
@btime enumAcc2(Sc, IA_L,  N) |> collect;
@btime enumAcc(Sc, IA_Li, N) |> collect;
@btime enumAcc2(Sc, IA_Li, N) |> collect;
@btime enumAcc(Sc, IA_B,  N) |> collect;
@btime enumAcc2(Sc, IA_B,  N) |> collect;
@btime enumAcc(Sc, IA_Bi, N) |> collect;
@btime enumAcc2(Sc, IA_Bi, N) |> collect;
@btime enumAcc(Sc, IA_E,  N) |> collect;
@btime enumAcc2(Sc, IA_E,  N) |> collect;
@btime enumAcc(Sc, IA_Ei, N) |> collect;
@btime enumAcc2(Sc, IA_Ei, N) |> collect;
@btime enumAcc(Sc, IA_D,  N) |> collect;
@btime enumAcc2(Sc, IA_D,  N) |> collect;
@btime enumAcc(Sc, IA_Di, N) |> collect;
@btime enumAcc2(Sc, IA_Di, N) |> collect;
@btime enumAcc(Sc, IA_O,  N) |> collect;
@btime enumAcc2(Sc, IA_O,  N) |> collect;
@btime enumAcc(Sc, IA_Oi, N) |> collect;
@btime enumAcc2(Sc, IA_Oi, N) |> collect;

#####
julia> S = [Interval(5,10), Interval(6,20), Interval(5,30), Interval(5,10), Interval(5,10), Interval(50,70), Interval(70,90), Interval(1,60), Interval(40,80)]

julia> @btime enumAcc(S, IA_A,  100) |> collect;
  199.967 μs (2592 allocations: 108.16 KiB)

julia> @btime enumAcc2(S, IA_A,  100) |> collect;
  330.145 μs (4830 allocations: 255.95 KiB)

julia> @btime enumAcc(S, IA_Ai, 100) |> collect;

  87.076 μs (941 allocations: 39.48 KiB)

julia> @btime enumAcc2(S, IA_Ai, 100) |> collect;
  167.021 μs (2402 allocations: 133.16 KiB)

julia> @btime enumAcc(S, IA_L,  100) |> collect;
  68.269 μs (4024 allocations: 190.11 KiB)

julia> @btime enumAcc2(S, IA_L,  100) |> collect;
  68.678 μs (4024 allocations: 190.11 KiB)

julia> @btime enumAcc(S, IA_Li, 100) |> collect;
  42.898 μs (2365 allocations: 138.27 KiB)

julia> @btime enumAcc2(S, IA_Li, 100) |> collect;
  42.365 μs (2365 allocations: 138.27 KiB)

julia> @btime enumAcc(S, IA_B,  100) |> collect;
  106.534 μs (1174 allocations: 60.95 KiB)

julia> @btime enumAcc2(S, IA_B,  100) |> collect;
  179.402 μs (2422 allocations: 135.33 KiB)

julia> @btime enumAcc(S, IA_Bi, 100) |> collect;
  197.658 μs (2474 allocations: 104.13 KiB)

julia> @btime enumAcc2(S, IA_Bi, 100) |> collect;
  297.958 μs (4428 allocations: 232.23 KiB)

julia> @btime enumAcc(S, IA_E,  100) |> collect;
  116.312 μs (1234 allocations: 62.02 KiB)

julia> @btime enumAcc2(S, IA_E,  100) |> collect;
  182.095 μs (2483 allocations: 137.66 KiB)

julia> @btime enumAcc(S, IA_Ei, 100) |> collect;
  97.437 μs (949 allocations: 39.73 KiB)

julia> @btime enumAcc2(S, IA_Ei, 100) |> collect;
  171.409 μs (2169 allocations: 112.92 KiB)

julia> @btime enumAcc(S, IA_D,  100) |> collect;
  1.477 ms (24592 allocations: 1.54 MiB)

julia> @btime enumAcc2(S, IA_D,  100) |> collect;
  2.555 ms (38518 allocations: 2.18 MiB)

julia> @btime enumAcc(S, IA_Di, 100) |> collect;
  1.629 ms (29874 allocations: 1.75 MiB)

julia> @btime enumAcc2(S, IA_Di, 100) |> collect;
  2.353 ms (42234 allocations: 2.30 MiB)

julia> @btime enumAcc(S, IA_O,  100) |> collect;
  3.097 ms (57419 allocations: 3.34 MiB)

julia> @btime enumAcc2(S, IA_O,  100) |> collect;
  4.549 ms (81039 allocations: 4.41 MiB)

julia> @btime enumAcc(S, IA_Oi, 100) |> collect;
  2.052 ms (35941 allocations: 2.21 MiB)

julia> @btime enumAcc2(S, IA_Oi, 100) |> collect;
  3.126 ms (53311 allocations: 2.99 MiB)

#####

=#
