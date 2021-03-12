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

################################################################################
# BEGIN Interval
################################################################################

struct Interval <: AbstractWorld
	x :: Integer
	y :: Integer
	# TODO check x<y but only in debug mode.  && x<=N, y<=N ?
	# Interval(x,y) = x>0 && y>0 && x < y ? new(x,y) : error("Can't instantiate Interval(x={$x},y={$y})")
	Interval(w::Interval) = new(w.x,w.y)
	Interval(x::Integer,y::Integer) = new(x,y)
	Interval((x,y)::Tuple{Integer,Integer}) = new(x,y)
	Interval(::_emptyWorld) = new(-1,0)
	Interval(::_firstWorld) = new(1,2)
	Interval(::_centeredWorld, X::Integer) = new(div(X,2)+1,div(X,2)+1+1+(isodd(X) ? 0 : 1))
end

show(io::IO, w::Interval) = begin
	print(io, "(")
	print(io, w.x)
	print(io, "−")
	print(io, w.y)
	print(io, ")")
end

# Convenience function: enumerate intervals in a given range
enumPairsIn(a::Integer, b::Integer) =
	Iterators.filter((a)->a[1]<a[2], Iterators.product(a:b-1, a+1:b)) # TODO try to avoid filter maybe


yieldReprs(test_operator::_TestOpGeq, repr::_ReprMax{Interval},  channel::MatricialChannel{T,1}) where {T} =
	reverse(extrema(readWorld(repr.w, channel)))::NTuple{2,T}
yieldReprs(test_operator::_TestOpGeq, repr::_ReprMin{Interval},  channel::MatricialChannel{T,1}) where {T} =
	extrema(readWorld(repr.w, channel))::NTuple{2,T}
yieldReprs(test_operator::_TestOpGeq, repr::_ReprVal{Interval},  channel::MatricialChannel{T,1}) where {T} =
	(channel[repr.w.x],channel[repr.w.x])::NTuple{2,T}
yieldReprs(test_operator::_TestOpGeq, repr::_ReprNone{Interval}, channel::MatricialChannel{T,1}) where {T} =
	(typemin(T),typemax(T))::NTuple{2,T}

yieldRepr(test_operator::Union{_TestOpGeq,_TestOpLes}, repr::_ReprMax{Interval},  channel::MatricialChannel{T,1}) where {T} =
	maximum(readWorld(repr.w, channel))::T
yieldRepr(test_operator::Union{_TestOpGeq,_TestOpLes}, repr::_ReprMin{Interval},  channel::MatricialChannel{T,1}) where {T} =
	minimum(readWorld(repr.w, channel))::T
yieldRepr(test_operator::Union{_TestOpGeq,_TestOpLes}, repr::_ReprVal{Interval},  channel::MatricialChannel{T,1}) where {T} =
	channel[repr.w.x]::T
yieldRepr(test_operator::_TestOpGeq, repr::_ReprNone{Interval}, channel::MatricialChannel{T,1}) where {T} =
	typemin(T)::T
yieldRepr(test_operator::_TestOpLes, repr::_ReprNone{Interval}, channel::MatricialChannel{T,1}) where {T} =
	typemax(T)::T

enumAccRepr(test_operator::_TestOpGeq, w::Interval, ::_RelationId,  X::Integer) = _ReprMin(w)
enumAccRepr(test_operator::_TestOpLes, w::Interval, ::_RelationId,  X::Integer) = _ReprMax(w)
enumAccRepr(test_operator::_TestOpGeq, w::Interval, ::_RelationAll, X::Integer) = _ReprMax(Interval(1,X+1))
enumAccRepr(test_operator::_TestOpLes, w::Interval, ::_RelationAll, X::Integer) = _ReprMin(Interval(1,X+1))

# TODO optimize relationAll
WExtremaModal(test_operator::_TestOpGeq, w::Interval, r::R where R<:AbstractRelation, channel::MatricialChannel{T,1}) where {T} =
	yieldReprs(test_operator, enumAccRepr(test_operator, w, r, size(channel)...), channel)
WExtremeModal(test_operator::Union{_TestOpGeq,_TestOpLes}, w::Interval, r::R where R<:AbstractRelation, channel::MatricialChannel{T,1}) where {T} =
	yieldRepr(test_operator, enumAccRepr(test_operator, w, r, size(channel)...), channel)



# TODO optimize relationAll?
# WExtremaModal(test_operator::_TestOpGeq, w::Interval, ::_RelationAll, channel::MatricialChannel{T,1}) where {T} = begin
# 	# X = length(channel)
# 	# println("Check!")
# 	# println(test_operator)
# 	# println(w)
# 	# println(relation)
# 	# println(channel)
# 	# println(WExtrema(test_operator, Interval(1,X+1), channel))
# 	# readline()
# 	# WExtrema(test_operator, Interval(1,X+1), channel)
# 	reverse(extrema(channel))
# end
# WExtremeModal(test_operator::_TestOpGeq, w::Interval, ::_RelationAll, channel::MatricialChannel{T,1}) where {T} = begin
# 	# TODO optimize this by replacing readworld with channel[1:X]...
# 	# X = length(channel)
# 	# maximum(readWorld(Interval(1,X+1),channel))
# 	maximum(channel)
# end
# WExtremeModal(test_operator::_TestOpLes, w::Interval, ::_RelationAll, channel::MatricialChannel{T,1}) where {T} = begin
# 	# TODO optimize this by replacing readworld with channel[1:X]...
# 	# X = length(channel)
# 	# minimum(readWorld(Interval(1,X+1),channel))
# 	minimum(channel)
# end
enumAccBare(w::Interval, ::_RelationId, XYZ::Vararg{Integer,N}) where N = [(w.x, w.y)]
enumAcc(S::AbstractWorldSet{Interval}, r::_RelationAll, X::Integer) =
	IterTools.imap(Interval, enumPairsIn(1, X+1))

worldTypeSize(::Type{Interval}) = 2
n_worlds(::Type{Interval}, channel_size::Tuple{Integer}) = div(channel_size[1]*(channel_size[1]+1),2)

print_world(w::Interval) = println("Interval [$(w.x),$(w.y)), length $(w.y-w.x)")

@inline readWorld(w::Interval, channel::MatricialChannel{T,1}) where {T} = channel[w.x:w.y-1]

################################################################################
# END Interval
################################################################################

################################################################################
# BEGIN IA relations
################################################################################

# Interval Algebra relations (or Allen relations)
abstract type _IARel <: AbstractRelation end
struct _IA_A  <: _IARel end; const IA_A  = _IA_A();  # After
struct _IA_L  <: _IARel end; const IA_L  = _IA_L();  # Later
struct _IA_B  <: _IARel end; const IA_B  = _IA_B();  # Begins
struct _IA_E  <: _IARel end; const IA_E  = _IA_E();  # Ends
struct _IA_D  <: _IARel end; const IA_D  = _IA_D();  # During
struct _IA_O  <: _IARel end; const IA_O  = _IA_O();  # Overlaps
struct _IA_Ai <: _IARel end; const IA_Ai = _IA_Ai(); # After inverse
struct _IA_Li <: _IARel end; const IA_Li = _IA_Li(); # Later inverse
struct _IA_Bi <: _IARel end; const IA_Bi = _IA_Bi(); # Begins inverse
struct _IA_Ei <: _IARel end; const IA_Ei = _IA_Ei(); # Ends inverse
struct _IA_Di <: _IARel end; const IA_Di = _IA_Di(); # During inverse
struct _IA_Oi <: _IARel end; const IA_Oi = _IA_Oi(); # Overlaps inverse

# TODO change name to display_rel_short
display_rel_short(::_IA_A)  = "A"
display_rel_short(::_IA_L)  = "L"
display_rel_short(::_IA_B)  = "B"
display_rel_short(::_IA_E)  = "E"
display_rel_short(::_IA_D)  = "D"
display_rel_short(::_IA_O)  = "O"
display_rel_short(::_IA_Ai) = "A̅"# "Ā"
display_rel_short(::_IA_Li) = "L̅"# "L̄"
display_rel_short(::_IA_Bi) = "B̅"# "B̄"
display_rel_short(::_IA_Ei) = "E̅"# "Ē"
display_rel_short(::_IA_Di) = "D̅"# "D̄"
display_rel_short(::_IA_Oi) = "O̅"# "Ō"

# 12 Interval Algebra relations
const IARelations = [IA_A,  IA_L,  IA_B,  IA_E,  IA_D,  IA_O,
										 IA_Ai, IA_Li, IA_Bi, IA_Ei, IA_Di, IA_Oi]

# 13 Interval Algebra extended with universal
const IARelations_extended = [RelationAll, IARelations]

# Enumerate accessible worlds from a single world
enumAccBare(w::Interval, ::_IA_A,  X::Integer) = zip(Iterators.repeated(w.y), w.y+1:X+1)
enumAccBare(w::Interval, ::_IA_Ai, X::Integer) = zip(1:w.x-1, Iterators.repeated(w.x))
enumAccBare(w::Interval, ::_IA_L,  X::Integer) = enumPairsIn(w.y+1, X+1)
enumAccBare(w::Interval, ::_IA_Li, X::Integer) = enumPairsIn(1, w.x-1)
enumAccBare(w::Interval, ::_IA_B,  X::Integer) = zip(Iterators.repeated(w.x), w.x+1:w.y-1)
enumAccBare(w::Interval, ::_IA_Bi, X::Integer) = zip(Iterators.repeated(w.x), w.y+1:X+1)
enumAccBare(w::Interval, ::_IA_E,  X::Integer) = zip(w.x+1:w.y-1, Iterators.repeated(w.y))
enumAccBare(w::Interval, ::_IA_Ei, X::Integer) = zip(1:w.x-1, Iterators.repeated(w.y))
enumAccBare(w::Interval, ::_IA_D,  X::Integer) = enumPairsIn(w.x+1, w.y-1)
enumAccBare(w::Interval, ::_IA_Di, X::Integer) = Iterators.product(1:w.x-1, w.y+1:X+1)
enumAccBare(w::Interval, ::_IA_O,  X::Integer) = Iterators.product(w.x+1:w.y-1, w.y+1:X+1)
enumAccBare(w::Interval, ::_IA_Oi, X::Integer) = Iterators.product(1:w.x-1, w.x+1:w.y-1)

# More efficient implementations for edge cases
enumAcc(S::AbstractWorldSet{Interval}, ::_IA_L, X::Integer) =
	IterTools.imap(Interval, enumAccBare(nth(S, argmin(map((w)->w.y, S))), IA_L, X))
enumAcc(S::AbstractWorldSet{Interval}, ::_IA_Li, X::Integer) =
	IterTools.imap(Interval, enumAccBare(nth(S, argmax(map((w)->w.x, S))), IA_Li, X))
enumAcc(S::AbstractWorldSet{Interval}, ::_IA_A, X::Integer) =
	IterTools.imap(Interval,
		Iterators.flatten(
			IterTools.imap((y)->zip(Iterators.repeated(y), y+1:X+1),
				IterTools.distinct(map((w)->w.y, S))
			)
		)
	)
enumAcc(S::AbstractWorldSet{Interval}, ::_IA_Ai, X::Integer) =
	IterTools.imap(Interval,
		Iterators.flatten(
			IterTools.imap((x)->zip(1:x-1, Iterators.repeated(x)),
				IterTools.distinct(map((w)->w.x, S))
			)
		)
	)

# Other options:
# enumAcc2_1_2(S::AbstractWorldSet{Interval}, ::_IA_L, X::Integer) =
# 	IterTools.imap(Interval, enumAccBare(Base.argmin((w.y for w in S)), IA_L, X))
# enumAcc2_1_2(S::AbstractWorldSet{Interval}, ::_IA_Li, X::Integer) =
# 	IterTools.imap(Interval, enumAccBare(Base.argmax((w.x for w in S)), IA_Li, X))
# enumAcc2_2(S::AbstractWorldSet{Interval}, ::_IA_L, X::Integer) = begin
# 	m = argmin(map((w)->w.y, S))
# 	IterTools.imap(Interval, enumAccBare([w for (i,w) in enumerate(S) if i == m][1], IA_L, X))
# end
# enumAcc2_2(S::AbstractWorldSet{Interval}, ::_IA_Li, X::Integer) = begin
# 	m = argmax(map((w)->w.x, S))
# 	IterTools.imap(Interval, enumAccBare([w for (i,w) in enumerate(S) if i == m][1], IA_Li, X))
# end
# # This makes sense if we have 2-Tuples instead of intervals
# function snd((a,b)::Tuple) b end
# function fst((a,b)::Tuple) a end
# enumAcc2_1(S::AbstractWorldSet{Interval}, ::_IA_L, X::Integer) = 
# 	IterTools.imap(Interval,
# 		enumAccBare(S[argmin(map(snd, S))], IA_L, X)
# 	)
# enumAcc2_1(S::AbstractWorldSet{Interval}, ::_IA_Li, X::Integer) = 
# 	IterTools.imap(Interval,
# 		enumAccBare(S[argmax(map(fst, S))], IA_Li, X)
# 	)



# IA_All max
# IA_Id  min
# -------
# IA_Bi  min
# IA_Ei  min
# IA_Di  min
# IA_O   min
# IA_Oi  min
# -------
# IA_L   max
# IA_Li  max
# IA_D   max
# -------
# IA_A   val
# IA_Ai  val
# IA_B   val
# IA_E   val

# TODO parametrize on the test_operator. These are wrong anyway...
# Note: these conditions are the ones that make a modalStep inexistent
enumAccRepr(test_operator::Union{_TestOpGeq,_TestOpLes}, w::Interval, ::_IA_A,  X::Integer) = (w.y < X+1)                 ? _ReprVal(Interval(w.y, w.y+1)   ) : _ReprNone{Interval}() # [Interval(w.y, X+1)]     : Interval[]
enumAccRepr(test_operator::Union{_TestOpGeq,_TestOpLes}, w::Interval, ::_IA_Ai, X::Integer) = (1 < w.x)                   ? _ReprVal(Interval(w.x-1, w.x)   ) : _ReprNone{Interval}() # [Interval(1, w.x)]       : Interval[]
enumAccRepr(test_operator::Union{_TestOpGeq,_TestOpLes}, w::Interval, ::_IA_B,  X::Integer) = (w.x < w.y-1)               ? _ReprVal(Interval(w.x, w.x+1)   ) : _ReprNone{Interval}() # [Interval(w.x, w.y-1)]   : Interval[]
enumAccRepr(test_operator::Union{_TestOpGeq,_TestOpLes}, w::Interval, ::_IA_E,  X::Integer) = (w.x+1 < w.y)               ? _ReprVal(Interval(w.y-1, w.y)   ) : _ReprNone{Interval}() # [Interval(w.x+1, w.y)]   : Interval[]

enumAccRepr(test_operator::_TestOpGeq, w::Interval, ::_IA_L,  X::Integer) = (w.y+1 < X+1)               ? _ReprMax(Interval(w.y+1, X+1)   ) : _ReprNone{Interval}() # [Interval(w.y+1, X+1)]   : Interval[]
enumAccRepr(test_operator::_TestOpGeq, w::Interval, ::_IA_Li, X::Integer) = (1 < w.x-1)                 ? _ReprMax(Interval(1, w.x-1)     ) : _ReprNone{Interval}() # [Interval(1, w.x-1)]     : Interval[]
enumAccRepr(test_operator::_TestOpGeq, w::Interval, ::_IA_D,  X::Integer) = (w.x+1 < w.y-1)             ? _ReprMax(Interval(w.x+1, w.y-1) ) : _ReprNone{Interval}() # [Interval(w.x+1, w.y-1)] : Interval[]
enumAccRepr(test_operator::_TestOpLes, w::Interval, ::_IA_L,  X::Integer) = (w.y+1 < X+1)               ? _ReprMin(Interval(w.y+1, X+1)   ) : _ReprNone{Interval}() # [Interval(w.y+1, X+1)]   : Interval[]
enumAccRepr(test_operator::_TestOpLes, w::Interval, ::_IA_Li, X::Integer) = (1 < w.x-1)                 ? _ReprMin(Interval(1, w.x-1)     ) : _ReprNone{Interval}() # [Interval(1, w.x-1)]     : Interval[]
enumAccRepr(test_operator::_TestOpLes, w::Interval, ::_IA_D,  X::Integer) = (w.x+1 < w.y-1)             ? _ReprMin(Interval(w.x+1, w.y-1) ) : _ReprNone{Interval}() # [Interval(w.x+1, w.y-1)] : Interval[]

enumAccRepr(test_operator::_TestOpGeq, w::Interval, ::_IA_Bi, X::Integer) = (w.y < X+1)                 ? _ReprMin(Interval(w.x, w.y+1)   ) : _ReprNone{Interval}() # [Interval(w.x, X+1)]     : Interval[]
enumAccRepr(test_operator::_TestOpGeq, w::Interval, ::_IA_Ei, X::Integer) = (1 < w.x)                   ? _ReprMin(Interval(w.x-1, w.y)   ) : _ReprNone{Interval}() # [Interval(1, w.y)]       : Interval[]
enumAccRepr(test_operator::_TestOpGeq, w::Interval, ::_IA_Di, X::Integer) = (1 < w.x && w.y < X+1)      ? _ReprMin(Interval(w.x-1, w.y+1) ) : _ReprNone{Interval}() # [Interval(1, X+1)]       : Interval[]
enumAccRepr(test_operator::_TestOpGeq, w::Interval, ::_IA_O,  X::Integer) = (w.x+1 < w.y && w.y < X+1)  ? _ReprMin(Interval(w.y-1, w.y+1) ) : _ReprNone{Interval}() # [Interval(w.x+1, X+1)]   : Interval[]
enumAccRepr(test_operator::_TestOpGeq, w::Interval, ::_IA_Oi, X::Integer) = (1 < w.x && w.x+1 < w.y)    ? _ReprMin(Interval(w.x-1, w.x+1) ) : _ReprNone{Interval}() # [Interval(1, w.y-1)]     : Interval[]
enumAccRepr(test_operator::_TestOpLes, w::Interval, ::_IA_Bi, X::Integer) = (w.y < X+1)                 ? _ReprMax(Interval(w.x, w.y+1)   ) : _ReprNone{Interval}() # [Interval(w.x, X+1)]     : Interval[]
enumAccRepr(test_operator::_TestOpLes, w::Interval, ::_IA_Ei, X::Integer) = (1 < w.x)                   ? _ReprMax(Interval(w.x-1, w.y)   ) : _ReprNone{Interval}() # [Interval(1, w.y)]       : Interval[]
enumAccRepr(test_operator::_TestOpLes, w::Interval, ::_IA_Di, X::Integer) = (1 < w.x && w.y < X+1)      ? _ReprMax(Interval(w.x-1, w.y+1) ) : _ReprNone{Interval}() # [Interval(1, X+1)]       : Interval[]
enumAccRepr(test_operator::_TestOpLes, w::Interval, ::_IA_O,  X::Integer) = (w.x+1 < w.y && w.y < X+1)  ? _ReprMax(Interval(w.y-1, w.y+1) ) : _ReprNone{Interval}() # [Interval(w.x+1, X+1)]   : Interval[]
enumAccRepr(test_operator::_TestOpLes, w::Interval, ::_IA_Oi, X::Integer) = (1 < w.x && w.x+1 < w.y)    ? _ReprMax(Interval(w.x-1, w.x+1) ) : _ReprNone{Interval}() # [Interval(1, w.y-1)]     : Interval[]

# WExtremaModal(test_operator::_TestOpGeq, w::Interval, ::_IA_A, channel::MatricialChannel{T,1}) where {T} =
# 	(w.y < length(channel)+1) ? (channel[w.y],channel[w.y]) : (typemax(T),typemin(T))
# WExtremeModal(test_operator::_TestOpGeq, w::Interval, ::_IA_A, channel::MatricialChannel{T,1}) where {T} =
# 	(w.y < length(channel)+1) ? channel[w.y] : typemax(T)
# WExtremeModal(test_operator::_TestOpLes, w::Interval, ::_IA_A, channel::MatricialChannel{T,1}) where {T} =
# 	(w.y < length(channel)+1) ? channel[w.y] : typemin(T)

# WExtremaModal(test_operator::_TestOpGeq, w::Interval, ::_IA_Ai, channel::MatricialChannel{T,1}) where {T} =
# 	(1 < w.x) ? (channel[w.x-1],channel[w.x-1]) : (typemax(T),typemin(T))
# WExtremeModal(test_operator::_TestOpGeq, w::Interval, ::_IA_Ai, channel::MatricialChannel{T,1}) where {T} =
# 	(1 < w.x) ? channel[w.x-1] : typemax(T)
# WExtremeModal(test_operator::_TestOpLes, w::Interval, ::_IA_Ai, channel::MatricialChannel{T,1}) where {T} =
# 	(1 < w.x) ? channel[w.x-1] : typemin(T)

# WExtremaModal(test_operator::_TestOpGeq, w::Interval, ::_IA_L, channel::MatricialChannel{T,1}) where {T} =
# 	(w.y+1 < length(channel)+1) ? reverse(extrema(channel[w.y+1:length(channel)])) : (typemax(T),typemin(T))
# WExtremeModal(test_operator::_TestOpGeq, w::Interval, ::_IA_L, channel::MatricialChannel{T,1}) where {T} =
# 	(w.y+1 < length(channel)+1) ? maximum(channel[w.y+1:length(channel)]) : typemax(T)
# WExtremeModal(test_operator::_TestOpLes, w::Interval, ::_IA_L, channel::MatricialChannel{T,1}) where {T} =
# 	(w.y+1 < length(channel)+1) ? minumum(channel[w.y+1:length(channel)]) : typemin(T)

# WExtremaModal(test_operator::_TestOpGeq, w::Interval, ::_IA_Li, channel::MatricialChannel{T,1}) where {T} =
# 	(1 < w.x-1) ? reverse(extrema(channel[1:w.x-2])) : (typemax(T),typemin(T))
# WExtremeModal(test_operator::_TestOpGeq, w::Interval, ::_IA_Li, channel::MatricialChannel{T,1}) where {T} =
# 	(1 < w.x-1) ? maximum(channel[1:w.x-2]) : typemax(T)
# WExtremeModal(test_operator::_TestOpLes, w::Interval, ::_IA_Li, channel::MatricialChannel{T,1}) where {T} =
# 	(1 < w.x-1) ? minumum(channel[1:w.x-2]) : typemin(T)

# WExtremaModal(test_operator::_TestOpGeq, w::Interval, ::_IA_B, channel::MatricialChannel{T,1}) where {T} =
# 	(w.x < w.y-1) ? (channel[w.x],channel[w.x]) : (typemax(T),typemin(T))
# WExtremeModal(test_operator::_TestOpGeq, w::Interval, ::_IA_B, channel::MatricialChannel{T,1}) where {T} =
# 	(w.x < w.y-1) ? channel[w.x] : typemax(T)
# WExtremeModal(test_operator::_TestOpLes, w::Interval, ::_IA_B, channel::MatricialChannel{T,1}) where {T} =
# 	(w.x < w.y-1) ? channel[w.x] : typemin(T)

# WExtremaModal(test_operator::_TestOpGeq, w::Interval, ::_IA_Bi, channel::MatricialChannel{T,1}) where {T} =
# 	(w.y < length(channel)+1) ? (minimum(channel[w.x:w.y-1+1]),maximum(channel[w.x:w.y-1+1])) : (typemax(T),typemin(T))
# WExtremeModal(test_operator::_TestOpGeq, w::Interval, ::_IA_Bi, channel::MatricialChannel{T,1}) where {T} =
# 	(w.y < length(channel)+1) ? minimum(channel[w.x:w.y-1+1]) : typemax(T)
# WExtremeModal(test_operator::_TestOpLes, w::Interval, ::_IA_Bi, channel::MatricialChannel{T,1}) where {T} =
# 	(w.y < length(channel)+1) ? maximum(channel[w.x:w.y-1+1]) : typemin(T)

# WExtremaModal(test_operator::_TestOpGeq, w::Interval, ::_IA_E, channel::MatricialChannel{T,1}) where {T} =
# 	(w.x+1 < w.y) ? (channel[w.y-1],channel[w.y-1]) : (typemax(T),typemin(T))
# WExtremeModal(test_operator::_TestOpGeq, w::Interval, ::_IA_E, channel::MatricialChannel{T,1}) where {T} =
# 	(w.x+1 < w.y) ? channel[w.y-1] : typemax(T)
# WExtremeModal(test_operator::_TestOpLes, w::Interval, ::_IA_E, channel::MatricialChannel{T,1}) where {T} =
# 	(w.x+1 < w.y) ? channel[w.y-1] : typemin(T)

# WExtremaModal(test_operator::_TestOpGeq, w::Interval, ::_IA_Ei, channel::MatricialChannel{T,1}) where {T} =
# 	(1 < w.x) ? (minimum(channel[w.x-1:w.y-1]),maximum(channel[w.x-1:w.y-1])) : (typemax(T),typemin(T))
# WExtremeModal(test_operator::_TestOpGeq, w::Interval, ::_IA_Ei, channel::MatricialChannel{T,1}) where {T} =
# 	(1 < w.x) ? minimum(channel[w.x-1:w.y-1]) : typemax(T)
# WExtremeModal(test_operator::_TestOpLes, w::Interval, ::_IA_Ei, channel::MatricialChannel{T,1}) where {T} =
# 	(1 < w.x) ? maximum(channel[w.x-1:w.y-1]) : typemin(T)

# WExtremaModal(test_operator::_TestOpGeq, w::Interval, ::_IA_D, channel::MatricialChannel{T,1}) where {T} =
# 	(w.x+1 < w.y-1) ? reverse(extrema(channel[w.x+1:w.y-1-1])) : (typemax(T),typemin(T))
# WExtremeModal(test_operator::_TestOpGeq, w::Interval, ::_IA_D, channel::MatricialChannel{T,1}) where {T} =
# 	(w.x+1 < w.y-1) ? maximum(channel[w.x+1:w.y-1-1]) : typemax(T)
# WExtremeModal(test_operator::_TestOpLes, w::Interval, ::_IA_D, channel::MatricialChannel{T,1}) where {T} =
# 	(w.x+1 < w.y-1) ? minumum(channel[w.x+1:w.y-1-1]) : typemin(T)

# WExtremaModal(test_operator::_TestOpGeq, w::Interval, ::_IA_Di, channel::MatricialChannel{T,1}) where {T} =
# 	(1 < w.x && w.y < length(channel)+1) ? (minimum(channel[w.x-1:w.y-1+1]),maximum(channel[w.x-1:w.y-1+1])) : (typemax(T),typemin(T))
# WExtremeModal(test_operator::_TestOpGeq, w::Interval, ::_IA_Di, channel::MatricialChannel{T,1}) where {T} =
# 	(1 < w.x && w.y < length(channel)+1) ? minimum(channel[w.x-1:w.y-1+1]) : typemax(T)
# WExtremeModal(test_operator::_TestOpLes, w::Interval, ::_IA_Di, channel::MatricialChannel{T,1}) where {T} =
# 	(1 < w.x && w.y < length(channel)+1) ? maximum(channel[w.x-1:w.y-1+1]) : typemin(T)

# WExtremaModal(test_operator::_TestOpGeq, w::Interval, ::_IA_O, channel::MatricialChannel{T,1}) where {T} =
# 	(w.x+1 < w.y && w.y < length(channel)+1) ? (minimum(channel[w.y-1:w.y-1+1]),maximum(channel[w.y-1:w.y-1+1])) : (typemax(T),typemin(T))
# WExtremeModal(test_operator::_TestOpGeq, w::Interval, ::_IA_O, channel::MatricialChannel{T,1}) where {T} =
# 	(w.x+1 < w.y && w.y < length(channel)+1) ? minimum(channel[w.y-1:w.y-1+1]) : typemax(T)
# WExtremeModal(test_operator::_TestOpLes, w::Interval, ::_IA_O, channel::MatricialChannel{T,1}) where {T} =
# 	(w.x+1 < w.y && w.y < length(channel)+1) ? maximum(channel[w.y-1:w.y-1+1]) : typemin(T)

# WExtremaModal(test_operator::_TestOpGeq, w::Interval, ::_IA_Oi, channel::MatricialChannel{T,1}) where {T} =
# 	(1 < w.x && w.x+1 < w.y) ? (minimum(channel[w.x-1:w.x]),maximum(channel[w.x-1:w.x])) : (typemax(T),typemin(T))
# WExtremeModal(test_operator::_TestOpGeq, w::Interval, ::_IA_Oi, channel::MatricialChannel{T,1}) where {T} =
# 	(1 < w.x && w.x+1 < w.y) ? minimum(channel[w.x-1:w.x]) : typemax(T)
# WExtremeModal(test_operator::_TestOpLes, w::Interval, ::_IA_Oi, channel::MatricialChannel{T,1}) where {T} =
# 	(1 < w.x && w.x+1 < w.y) ? maximum(channel[w.x-1:w.x]) : typemin(T)

################################################################################
# END IA relations
################################################################################

################################################################################
# BEGIN Interval2D
################################################################################

# TODO generalize this as NTuple{4,Interval} ?
# 2-dimensional Interval counterpart: combination of two Intervals 
struct Interval2D <: AbstractWorld
	x :: Interval
	y :: Interval
	Interval2D(w::Interval2D) = new(w.x,w.y)
	Interval2D(x::Interval,y::Interval) = new(x,y)
	Interval2D(((xx,xy),(yx,yy))::Tuple{Tuple{Integer,Integer},Tuple{Integer,Integer}}) = new(Interval(xx,xy),Interval(yx,yy))
	Interval2D((x,y)::Tuple{Interval,Interval}) = new(x,y)
	Interval2D(x::Tuple{Integer,Integer}, y::Tuple{Integer,Integer}) = new(Interval(x),Interval(y))
	Interval2D(w::_emptyWorld) = new(Interval(w),Interval(w))
	Interval2D(w::_firstWorld) = new(Interval(w),Interval(w))
	Interval2D(w::_centeredWorld, X::Integer, Y::Integer) = new(Interval(w,X),Interval(w,Y))
end

show(io::IO, w::Interval2D) = begin
	print(io, "(")
	print(io, w.x)
	print(io, "×")
	print(io, w.y)
	print(io, ")")
end

yieldReprs(test_operator::_TestOpGeq, repr::_ReprMax{Interval2D},  channel::MatricialChannel{T,2}) where {T} =
	reverse(extrema(readWorld(repr.w, channel)))::NTuple{2,T}
yieldReprs(test_operator::_TestOpGeq, repr::_ReprMin{Interval2D},  channel::MatricialChannel{T,2}) where {T} =
	extrema(readWorld(repr.w, channel))::NTuple{2,T}
yieldReprs(test_operator::_TestOpGeq, repr::_ReprVal{Interval2D},  channel::MatricialChannel{T,2}) where {T} =
	(channel[repr.w.x.x, repr.w.y.x],channel[repr.w.x.x, repr.w.y.x])::NTuple{2,T}
yieldReprs(test_operator::_TestOpGeq, repr::_ReprNone{Interval2D}, channel::MatricialChannel{T,2}) where {T} =
	(typemin(T),typemax(T))::NTuple{2,T}

yieldRepr(test_operator::Union{_TestOpGeq,_TestOpLes}, repr::_ReprMax{Interval2D},  channel::MatricialChannel{T,2}) where {T} =
	maximum(readWorld(repr.w, channel))::T
yieldRepr(test_operator::Union{_TestOpGeq,_TestOpLes}, repr::_ReprMin{Interval2D},  channel::MatricialChannel{T,2}) where {T} =
	minimum(readWorld(repr.w, channel))::T
yieldRepr(test_operator::Union{_TestOpGeq,_TestOpLes}, repr::_ReprVal{Interval2D},  channel::MatricialChannel{T,2}) where {T} =
	channel[repr.w.x.x, repr.w.y.x]::T
yieldRepr(test_operator::_TestOpGeq, repr::_ReprNone{Interval2D}, channel::MatricialChannel{T,2}) where {T} =
	typemin(T)::T
yieldRepr(test_operator::_TestOpLes, repr::_ReprNone{Interval2D}, channel::MatricialChannel{T,2}) where {T} =
	typemax(T)::T

enumAccRepr(test_operator::_TestOpGeq, w::Interval2D, ::_RelationId,  X::Integer, Y::Integer) = _ReprMin(w)
enumAccRepr(test_operator::_TestOpLes, w::Interval2D, ::_RelationId,  X::Integer, Y::Integer) = _ReprMax(w)
enumAccRepr(test_operator::_TestOpGeq, w::Interval2D, ::_RelationAll, X::Integer, Y::Integer) = _ReprMax(Interval2D(Interval(1,X+1), Interval(1,Y+1)))
enumAccRepr(test_operator::_TestOpLes, w::Interval2D, ::_RelationAll, X::Integer, Y::Integer) = _ReprMin(Interval2D(Interval(1,X+1), Interval(1,Y+1)))

# TODO write only one ExtremeModal/ExtremaModal
# TODO optimize relationAll
WExtremaModal(test_operator::_TestOpGeq, w::Interval2D, r::R where R<:AbstractRelation, channel::MatricialChannel{T,2}) where {T} = begin
	# if (channel == [412 489 559 619 784; 795 771 1317 854 1256; 971 874 878 1278 560] && w.x.x==1 && w.x.y==3 && w.y.x==3 && w.y.y==4)
	# 	println(enumAccRepr(test_operator, w, r, size(channel)...))
	# 	readline()
	# end
	yieldReprs(test_operator, enumAccRepr(test_operator, w, r, size(channel)...), channel)
end
WExtremeModal(test_operator::Union{_TestOpGeq,_TestOpLes}, w::Interval2D, r::R where R<:AbstractRelation, channel::MatricialChannel{T,2}) where {T} =
	yieldRepr(test_operator, enumAccRepr(test_operator, w, r, size(channel)...), channel)
# channel = [1,2,3,2,8,349,0,830,7290,298,20,29,2790,27,90279,270,2722,79072,0]
# w = ModalLogic.Interval(3,9)
# # w = ModalLogic.Interval(3,4)
# for relation in ModalLogic.IARelations
# 	ModalLogic.WExtremaModal(ModalLogic.TestOpGeq, w, relation, channel)
# end

# channel2 = randn(3,4)
# channel2[1:3,1]
# channel2[1:3,2]
# channel2[1:3,3]
# channel2[1:3,4]
# vals=channel2
# mapslices(maximum, vals, dims=1)

# WExtremaModal(test_operator::_TestOpGeq, w::Interval2D, ::_RelationAll, channel::MatricialChannel{T,2}) where {T} = begin
# 	# X = size(channel, 1)
# 	# Y = size(channel, 2)
# 	# println("Check!")
# 	# println(test_operator)
# 	# println(w)
# 	# println(relation)
# 	# println(channel)
# 	# println(WExtrema(test_operator, Interval2D(Interval(1,X+1), Interval(1, Y+1)), channel))
# 	# readline()
# 	# WExtrema(test_operator, Interval2D(Interval(1,X+1), Interval(1, Y+1)), channel)
# 	reverse(extrema(channel))
# end
# WExtremeModal(test_operator::_TestOpGeq, w::Interval2D, ::_RelationAll, channel::MatricialChannel{T,2}) where {T} = begin
# 	# TODO optimize this by replacing readworld with channel[1:X]...
# 	# X = size(channel, 1)
# 	# Y = size(channel, 2)
# 	# maximum(channel[1:X,1:Y])
# 	maximum(channel)
# end
# WExtremeModal(test_operator::_TestOpLes, w::Interval2D, ::_RelationAll, channel::MatricialChannel{T,2}) where {T} = begin
# 	# TODO optimize this by replacing readworld with channel[1:X]...
# 	# X = size(channel, 1)
# 	# Y = size(channel, 2)
# 	# println(channel)
# 	# println(w)
# 	# println(minimum(channel[1:X,1:Y]))
# 	# readline()
# 	# minimum(channel[1:X,1:Y])
# 	minimum(channel)
# end
# enumAccBare(w::Interval2D, ::_RelationId, XYZ::Vararg{Integer,N}) where N = [(w.x, w.y)]
enumAcc(S::AbstractWorldSet{Interval2D}, r::_RelationAll, X::Integer, Y::Integer) =
	IterTools.imap(Interval2D,
		Iterators.product(enumPairsIn(1, X+1), enumPairsIn(1, Y+1))
		# enumAccBare(w..., IA2DRel(RelationAll,RelationAll), X, Y)
	)
	# IterTools.imap(Interval2D, enumPairsIn(1, X+1), enumPairsIn(1, Y+1))
		# enumAccBare(w, IA2DRel(RelationAll,RelationAll), X, Y)
# enumAccBare(w::Interval2D, r::_RelationAll, X::Integer, Y::Integer) =
# 	enumAccBare(w, _IA2DRel(RelationAll,RelationAll), X, Y)

worldTypeSize(::Type{Interval2D}) = 4
n_worlds(::Type{Interval2D}, channel_size::Tuple{Integer,Integer}) = n_worlds(Interval, channel_size[1]) * n_worlds(Interval, channel_size[2])

print_world(w::Interval2D) = println("Interval2D [$(w.x.x),$(w.x.y)) × [$(w.y.x),$(w.y.y)), length $(w.x.y-w.x.x)×$(w.y.y-w.y.x) = $((w.x.y-w.x.x)*(w.y.y-w.y.x))")

@inline readWorld(w::Interval2D, channel::MatricialChannel{T,2}) where {T} = channel[w.x.x:w.x.y-1,w.y.x:w.y.y-1]

################################################################################
# END Interval2D
################################################################################

################################################################################
# BEGIN IA2D relations
################################################################################

# 2D Interval Algebra relations (or Rectangle Algebra relations)
# Defined as combination of Interval Algebra, RelationId & RelationAll
const _IABase = Union{_IARel,_RelationId,_RelationAll}
struct _IA2DRel{R1<:_IABase,R2<:_IABase} <: AbstractRelation
	x :: R1
	y :: R2
end

# Interval Algebra relations
                                                   const IA_IU  = _IA2DRel(RelationId  , RelationAll); const IA_IA  = _IA2DRel(RelationId  , IA_A); const IA_IL  = _IA2DRel(RelationId  , IA_L); const IA_IB  = _IA2DRel(RelationId  , IA_B); const IA_IE  = _IA2DRel(RelationId  , IA_E); const IA_ID  = _IA2DRel(RelationId  , IA_D); const IA_IO  = _IA2DRel(RelationId  , IA_O); const IA_IAi  = _IA2DRel(RelationId  , IA_Ai); const IA_ILi  = _IA2DRel(RelationId  , IA_Li); const IA_IBi  = _IA2DRel(RelationId  , IA_Bi); const IA_IEi  = _IA2DRel(RelationId  , IA_Ei); const IA_IDi  = _IA2DRel(RelationId  , IA_Di); const IA_IOi  = _IA2DRel(RelationId  , IA_Oi);
const IA_UI  = _IA2DRel(RelationAll , RelationId);                                                     const IA_UA  = _IA2DRel(RelationAll , IA_A); const IA_UL  = _IA2DRel(RelationAll , IA_L); const IA_UB  = _IA2DRel(RelationAll , IA_B); const IA_UE  = _IA2DRel(RelationAll , IA_E); const IA_UD  = _IA2DRel(RelationAll , IA_D); const IA_UO  = _IA2DRel(RelationAll , IA_O); const IA_UAi  = _IA2DRel(RelationAll , IA_Ai); const IA_ULi  = _IA2DRel(RelationAll , IA_Li); const IA_UBi  = _IA2DRel(RelationAll , IA_Bi); const IA_UEi  = _IA2DRel(RelationAll , IA_Ei); const IA_UDi  = _IA2DRel(RelationAll , IA_Di); const IA_UOi  = _IA2DRel(RelationAll , IA_Oi);
const IA_AI  = _IA2DRel(IA_A        , RelationId); const IA_AU  = _IA2DRel(IA_A        , RelationAll); const IA_AA  = _IA2DRel(IA_A        , IA_A); const IA_AL  = _IA2DRel(IA_A        , IA_L); const IA_AB  = _IA2DRel(IA_A        , IA_B); const IA_AE  = _IA2DRel(IA_A        , IA_E); const IA_AD  = _IA2DRel(IA_A        , IA_D); const IA_AO  = _IA2DRel(IA_A        , IA_O); const IA_AAi  = _IA2DRel(IA_A        , IA_Ai); const IA_ALi  = _IA2DRel(IA_A        , IA_Li); const IA_ABi  = _IA2DRel(IA_A        , IA_Bi); const IA_AEi  = _IA2DRel(IA_A        , IA_Ei); const IA_ADi  = _IA2DRel(IA_A        , IA_Di); const IA_AOi  = _IA2DRel(IA_A        , IA_Oi);
const IA_LI  = _IA2DRel(IA_L        , RelationId); const IA_LU  = _IA2DRel(IA_L        , RelationAll); const IA_LA  = _IA2DRel(IA_L        , IA_A); const IA_LL  = _IA2DRel(IA_L        , IA_L); const IA_LB  = _IA2DRel(IA_L        , IA_B); const IA_LE  = _IA2DRel(IA_L        , IA_E); const IA_LD  = _IA2DRel(IA_L        , IA_D); const IA_LO  = _IA2DRel(IA_L        , IA_O); const IA_LAi  = _IA2DRel(IA_L        , IA_Ai); const IA_LLi  = _IA2DRel(IA_L        , IA_Li); const IA_LBi  = _IA2DRel(IA_L        , IA_Bi); const IA_LEi  = _IA2DRel(IA_L        , IA_Ei); const IA_LDi  = _IA2DRel(IA_L        , IA_Di); const IA_LOi  = _IA2DRel(IA_L        , IA_Oi);
const IA_BI  = _IA2DRel(IA_B        , RelationId); const IA_BU  = _IA2DRel(IA_B        , RelationAll); const IA_BA  = _IA2DRel(IA_B        , IA_A); const IA_BL  = _IA2DRel(IA_B        , IA_L); const IA_BB  = _IA2DRel(IA_B        , IA_B); const IA_BE  = _IA2DRel(IA_B        , IA_E); const IA_BD  = _IA2DRel(IA_B        , IA_D); const IA_BO  = _IA2DRel(IA_B        , IA_O); const IA_BAi  = _IA2DRel(IA_B        , IA_Ai); const IA_BLi  = _IA2DRel(IA_B        , IA_Li); const IA_BBi  = _IA2DRel(IA_B        , IA_Bi); const IA_BEi  = _IA2DRel(IA_B        , IA_Ei); const IA_BDi  = _IA2DRel(IA_B        , IA_Di); const IA_BOi  = _IA2DRel(IA_B        , IA_Oi);
const IA_EI  = _IA2DRel(IA_E        , RelationId); const IA_EU  = _IA2DRel(IA_E        , RelationAll); const IA_EA  = _IA2DRel(IA_E        , IA_A); const IA_EL  = _IA2DRel(IA_E        , IA_L); const IA_EB  = _IA2DRel(IA_E        , IA_B); const IA_EE  = _IA2DRel(IA_E        , IA_E); const IA_ED  = _IA2DRel(IA_E        , IA_D); const IA_EO  = _IA2DRel(IA_E        , IA_O); const IA_EAi  = _IA2DRel(IA_E        , IA_Ai); const IA_ELi  = _IA2DRel(IA_E        , IA_Li); const IA_EBi  = _IA2DRel(IA_E        , IA_Bi); const IA_EEi  = _IA2DRel(IA_E        , IA_Ei); const IA_EDi  = _IA2DRel(IA_E        , IA_Di); const IA_EOi  = _IA2DRel(IA_E        , IA_Oi);
const IA_DI  = _IA2DRel(IA_D        , RelationId); const IA_DU  = _IA2DRel(IA_D        , RelationAll); const IA_DA  = _IA2DRel(IA_D        , IA_A); const IA_DL  = _IA2DRel(IA_D        , IA_L); const IA_DB  = _IA2DRel(IA_D        , IA_B); const IA_DE  = _IA2DRel(IA_D        , IA_E); const IA_DD  = _IA2DRel(IA_D        , IA_D); const IA_DO  = _IA2DRel(IA_D        , IA_O); const IA_DAi  = _IA2DRel(IA_D        , IA_Ai); const IA_DLi  = _IA2DRel(IA_D        , IA_Li); const IA_DBi  = _IA2DRel(IA_D        , IA_Bi); const IA_DEi  = _IA2DRel(IA_D        , IA_Ei); const IA_DDi  = _IA2DRel(IA_D        , IA_Di); const IA_DOi  = _IA2DRel(IA_D        , IA_Oi);
const IA_OI  = _IA2DRel(IA_O        , RelationId); const IA_OU  = _IA2DRel(IA_O        , RelationAll); const IA_OA  = _IA2DRel(IA_O        , IA_A); const IA_OL  = _IA2DRel(IA_O        , IA_L); const IA_OB  = _IA2DRel(IA_O        , IA_B); const IA_OE  = _IA2DRel(IA_O        , IA_E); const IA_OD  = _IA2DRel(IA_O        , IA_D); const IA_OO  = _IA2DRel(IA_O        , IA_O); const IA_OAi  = _IA2DRel(IA_O        , IA_Ai); const IA_OLi  = _IA2DRel(IA_O        , IA_Li); const IA_OBi  = _IA2DRel(IA_O        , IA_Bi); const IA_OEi  = _IA2DRel(IA_O        , IA_Ei); const IA_ODi  = _IA2DRel(IA_O        , IA_Di); const IA_OOi  = _IA2DRel(IA_O        , IA_Oi);
const IA_AiI = _IA2DRel(IA_Ai       , RelationId); const IA_AiU = _IA2DRel(IA_Ai       , RelationAll); const IA_AiA = _IA2DRel(IA_Ai       , IA_A); const IA_AiL = _IA2DRel(IA_Ai       , IA_L); const IA_AiB = _IA2DRel(IA_Ai       , IA_B); const IA_AiE = _IA2DRel(IA_Ai       , IA_E); const IA_AiD = _IA2DRel(IA_Ai       , IA_D); const IA_AiO = _IA2DRel(IA_Ai       , IA_O); const IA_AiAi = _IA2DRel(IA_Ai       , IA_Ai); const IA_AiLi = _IA2DRel(IA_Ai       , IA_Li); const IA_AiBi = _IA2DRel(IA_Ai       , IA_Bi); const IA_AiEi = _IA2DRel(IA_Ai       , IA_Ei); const IA_AiDi = _IA2DRel(IA_Ai       , IA_Di); const IA_AiOi = _IA2DRel(IA_Ai       , IA_Oi);
const IA_LiI = _IA2DRel(IA_Li       , RelationId); const IA_LiU = _IA2DRel(IA_Li       , RelationAll); const IA_LiA = _IA2DRel(IA_Li       , IA_A); const IA_LiL = _IA2DRel(IA_Li       , IA_L); const IA_LiB = _IA2DRel(IA_Li       , IA_B); const IA_LiE = _IA2DRel(IA_Li       , IA_E); const IA_LiD = _IA2DRel(IA_Li       , IA_D); const IA_LiO = _IA2DRel(IA_Li       , IA_O); const IA_LiAi = _IA2DRel(IA_Li       , IA_Ai); const IA_LiLi = _IA2DRel(IA_Li       , IA_Li); const IA_LiBi = _IA2DRel(IA_Li       , IA_Bi); const IA_LiEi = _IA2DRel(IA_Li       , IA_Ei); const IA_LiDi = _IA2DRel(IA_Li       , IA_Di); const IA_LiOi = _IA2DRel(IA_Li       , IA_Oi);
const IA_BiI = _IA2DRel(IA_Bi       , RelationId); const IA_BiU = _IA2DRel(IA_Bi       , RelationAll); const IA_BiA = _IA2DRel(IA_Bi       , IA_A); const IA_BiL = _IA2DRel(IA_Bi       , IA_L); const IA_BiB = _IA2DRel(IA_Bi       , IA_B); const IA_BiE = _IA2DRel(IA_Bi       , IA_E); const IA_BiD = _IA2DRel(IA_Bi       , IA_D); const IA_BiO = _IA2DRel(IA_Bi       , IA_O); const IA_BiAi = _IA2DRel(IA_Bi       , IA_Ai); const IA_BiLi = _IA2DRel(IA_Bi       , IA_Li); const IA_BiBi = _IA2DRel(IA_Bi       , IA_Bi); const IA_BiEi = _IA2DRel(IA_Bi       , IA_Ei); const IA_BiDi = _IA2DRel(IA_Bi       , IA_Di); const IA_BiOi = _IA2DRel(IA_Bi       , IA_Oi);
const IA_EiI = _IA2DRel(IA_Ei       , RelationId); const IA_EiU = _IA2DRel(IA_Ei       , RelationAll); const IA_EiA = _IA2DRel(IA_Ei       , IA_A); const IA_EiL = _IA2DRel(IA_Ei       , IA_L); const IA_EiB = _IA2DRel(IA_Ei       , IA_B); const IA_EiE = _IA2DRel(IA_Ei       , IA_E); const IA_EiD = _IA2DRel(IA_Ei       , IA_D); const IA_EiO = _IA2DRel(IA_Ei       , IA_O); const IA_EiAi = _IA2DRel(IA_Ei       , IA_Ai); const IA_EiLi = _IA2DRel(IA_Ei       , IA_Li); const IA_EiBi = _IA2DRel(IA_Ei       , IA_Bi); const IA_EiEi = _IA2DRel(IA_Ei       , IA_Ei); const IA_EiDi = _IA2DRel(IA_Ei       , IA_Di); const IA_EiOi = _IA2DRel(IA_Ei       , IA_Oi);
const IA_DiI = _IA2DRel(IA_Di       , RelationId); const IA_DiU = _IA2DRel(IA_Di       , RelationAll); const IA_DiA = _IA2DRel(IA_Di       , IA_A); const IA_DiL = _IA2DRel(IA_Di       , IA_L); const IA_DiB = _IA2DRel(IA_Di       , IA_B); const IA_DiE = _IA2DRel(IA_Di       , IA_E); const IA_DiD = _IA2DRel(IA_Di       , IA_D); const IA_DiO = _IA2DRel(IA_Di       , IA_O); const IA_DiAi = _IA2DRel(IA_Di       , IA_Ai); const IA_DiLi = _IA2DRel(IA_Di       , IA_Li); const IA_DiBi = _IA2DRel(IA_Di       , IA_Bi); const IA_DiEi = _IA2DRel(IA_Di       , IA_Ei); const IA_DiDi = _IA2DRel(IA_Di       , IA_Di); const IA_DiOi = _IA2DRel(IA_Di       , IA_Oi);
const IA_OiI = _IA2DRel(IA_Oi       , RelationId); const IA_OiU = _IA2DRel(IA_Oi       , RelationAll); const IA_OiA = _IA2DRel(IA_Oi       , IA_A); const IA_OiL = _IA2DRel(IA_Oi       , IA_L); const IA_OiB = _IA2DRel(IA_Oi       , IA_B); const IA_OiE = _IA2DRel(IA_Oi       , IA_E); const IA_OiD = _IA2DRel(IA_Oi       , IA_D); const IA_OiO = _IA2DRel(IA_Oi       , IA_O); const IA_OiAi = _IA2DRel(IA_Oi       , IA_Ai); const IA_OiLi = _IA2DRel(IA_Oi       , IA_Li); const IA_OiBi = _IA2DRel(IA_Oi       , IA_Bi); const IA_OiEi = _IA2DRel(IA_Oi       , IA_Ei); const IA_OiDi = _IA2DRel(IA_Oi       , IA_Di); const IA_OiOi = _IA2DRel(IA_Oi       , IA_Oi);

# Print 2D Interval Algebra relations
display_rel_short(::_IA2DRel{_XR,_YR}) where {_XR<:_IABase,_YR<:_IABase} =
	string(display_rel_short(_XR()), ",", display_rel_short(_YR())); 

# 13^2-1=168 Rectangle Algebra relations
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

# 2*13=26 const _IA2DRelations_U =
	# Union{_IA2DRel{_RelationAll,_RelationId},_IA2DRel{_RelationAll,_IA_A},_IA2DRel{_RelationAll,_IA_L},_IA2DRel{_RelationAll,_IA_B},_IA2DRel{_RelationAll,_IA_E},_IA2DRel{_RelationAll,_IA_D},_IA2DRel{_RelationAll,_IA_O},_IA2DRel{_RelationAll,_IA_Ai},_IA2DRel{_RelationAll,_IA_Li},_IA2DRel{_RelationAll,_IA_Bi},_IA2DRel{_RelationAll,_IA_Ei},_IA2DRel{_RelationAll,_IA_Di},_IA2DRel{_RelationAll,_IA_Oi},
				# _IA2DRel{_RelationId,_RelationAll},_IA2DRel{_IA_A,_RelationAll},_IA2DRel{_IA_L,_RelationAll},_IA2DRel{_IA_B,_RelationAll},_IA2DRel{_IA_E,_RelationAll},_IA2DRel{_IA_D,_RelationAll},_IA2DRel{_IA_O,_RelationAll},_IA2DRel{_IA_Ai,_IA_Ai},_IA2DRel{_IA_Li,_IA_Li},_IA2DRel{_IA_Bi,_IA_Bi},_IA2DRel{_IA_Ei,_IA_Ei},_IA2DRel{_IA_Di,_IA_Di},_IA2DRel{_IA_Oi,_IA_Oi}}
const IA2DRelations_U = [
IA_UI ,IA_UA ,IA_UL ,IA_UB ,IA_UE ,IA_UD ,IA_UO ,IA_UAi ,IA_ULi ,IA_UBi ,IA_UEi ,IA_UDi ,IA_UOi,
IA_IU ,IA_AU ,IA_LU ,IA_BU ,IA_EU ,IA_DU ,IA_OU ,IA_AiU ,IA_LiU ,IA_BiU ,IA_EiU ,IA_DiU ,IA_OiU
]

# 14^2-1=195 Rectangle Algebra relations extended with their combinations with universal
const IA2DRelations_extended = [
RelationAll,
IA2DRelations...,
IA2DRelations_U...
]

# Enumerate accessible worlds from a single world
# Any 2D Interval Algebra relation is computed by combining the two one-dimensional relations
# TODO check if dimensions are to be swapped
# TODO these three are weird, the problem is _RelationAll
enumAccBare2(w::Interval, r::R where R<:_IARel, X::Integer) = enumAccBare(w,r,X)
enumAccBare2(w::Interval, r::_RelationId, X::Integer) = enumAccBare(w,r,X)
enumAccBare2(w::Interval, r::_RelationAll, X::Integer) =
	enumPairsIn(1, X+1)
	# IterTools.imap(Interval, enumPairsIn(1, X+1))

enumAccBare(w::Interval2D, r::R where R<:_IA2DRel, X::Integer, Y::Integer) =
	Iterators.product(enumAccBare2(w.x, r.x, X), enumAccBare2(w.y, r.y, Y))
	# TODO try instead: Iterators.product(enumAcc(w.x, r.x, X), enumAcc(w.y, r.y, Y))

# More efficient implementations for edge cases
# TODO write efficient implementations for _IA2DRelations_U
# enumAcc(S::AbstractWorldSet{Interval2D}, r::_IA2DRelations_U, X::Integer, Y::Integer) = begin
# 	IterTools.imap(Interval2D,
# 		Iterators.flatten(
# 			Iterators.product((enumAcc(w, r.x, X) for w in S), enumAcc(S, r, Y))
# 		)
# 	)
# end

# enumAccRepr for _IA2DRelations_U
# 3 operator categories for the 13+1 relations
const _IA2DRelMaximizer = Union{_RelationAll,_IA_L,_IA_Li,_IA_D}
const _IA2DRelMinimizer = Union{_RelationId,_IA_O,_IA_Oi,_IA_Bi,_IA_Ei,_IA_Di}
const _IA2DRelSingleVal = Union{_IA_A,_IA_Ai,_IA_B,_IA_E}

@inline enumAccRepr2D(test_operator::_TestOpGeq, w::Interval2D, rx::R1 where R1<:AbstractRelation, ry::R2 where R2<:AbstractRelation, X::Integer, Y::Integer, _ReprConstructor::Type{rT}) where {rT<:_ReprTreatment} = begin
	x = enumAccRepr(test_operator, w.x, rx, X)
	# println(x)
	if x == _ReprNone{Interval}()
		return _ReprNone{Interval2D}()
	end
	y = enumAccRepr(test_operator, w.y, ry, Y)
	# println(y)
	if y == _ReprNone{Interval}()
		return _ReprNone{Interval2D}()
	end
	return _ReprConstructor(Interval2D(x.w, y.w))
end

# 3*3 = 9 cases ((13+1)^2 = 196 relations)
# Maximizer operators
enumAccRepr(test_operator::_TestOpGeq, w::Interval2D, r::_IA2DRel{R1,R2} where {R1<:_IA2DRelMaximizer,R2<:_IA2DRelMaximizer}, X::Integer, Y::Integer) =
	enumAccRepr2D(test_operator, w, r.x, r.y, X, Y, _ReprMax)
enumAccRepr(test_operator::_TestOpGeq, w::Interval2D, r::_IA2DRel{R1,R2} where {R1<:_IA2DRelMinimizer,R2<:_IA2DRelMinimizer}, X::Integer, Y::Integer) = begin
	# println(enumAccRepr2D(test_operator, w, r.x, r.y, X, Y, _ReprMin))
	enumAccRepr2D(test_operator, w, r.x, r.y, X, Y, _ReprMin)
end
enumAccRepr(test_operator::_TestOpGeq, w::Interval2D, r::_IA2DRel{R1,R2} where {R1<:_IA2DRelSingleVal,R2<:_IA2DRelSingleVal}, X::Integer, Y::Integer) =
	enumAccRepr2D(test_operator, w, r.x, r.y, X, Y, _ReprVal)

enumAccRepr(test_operator::_TestOpGeq, w::Interval2D, r::_IA2DRel{R1,R2} where {R1<:_IA2DRelMaximizer,R2<:_IA2DRelSingleVal}, X::Integer, Y::Integer) =
	enumAccRepr2D(test_operator, w, r.x, r.y, X, Y, _ReprMax)
enumAccRepr(test_operator::_TestOpGeq, w::Interval2D, r::_IA2DRel{R1,R2} where {R1<:_IA2DRelMinimizer,R2<:_IA2DRelSingleVal}, X::Integer, Y::Integer) =
	enumAccRepr2D(test_operator, w, r.x, r.y, X, Y, _ReprMin)
enumAccRepr(test_operator::_TestOpGeq, w::Interval2D, r::_IA2DRel{R1,R2} where {R1<:_IA2DRelSingleVal,R2<:_IA2DRelMaximizer}, X::Integer, Y::Integer) =
	enumAccRepr2D(test_operator, w, r.x, r.y, X, Y, _ReprMax)
enumAccRepr(test_operator::_TestOpGeq, w::Interval2D, r::_IA2DRel{R1,R2} where {R1<:_IA2DRelSingleVal,R2<:_IA2DRelMinimizer}, X::Integer, Y::Integer) =
	enumAccRepr2D(test_operator, w, r.x, r.y, X, Y, _ReprMin)

enumAccRepr(test_operator::_TestOpLes, w::Interval2D, r::_IA2DRel{R1,R2} where {R1<:_IA2DRelMaximizer,R2<:_IA2DRelMaximizer}, X::Integer, Y::Integer) =
	enumAccRepr2D(test_operator, w, r.x, r.y, X, Y, _ReprMin)
enumAccRepr(test_operator::_TestOpLes, w::Interval2D, r::_IA2DRel{R1,R2} where {R1<:_IA2DRelMinimizer,R2<:_IA2DRelMinimizer}, X::Integer, Y::Integer) =
	enumAccRepr2D(test_operator, w, r.x, r.y, X, Y, _ReprMax)
enumAccRepr(test_operator::_TestOpLes, w::Interval2D, r::_IA2DRel{R1,R2} where {R1<:_IA2DRelSingleVal,R2<:_IA2DRelSingleVal}, X::Integer, Y::Integer) =
	enumAccRepr2D(test_operator, w, r.x, r.y, X, Y, _ReprVal)

enumAccRepr(test_operator::_TestOpLes, w::Interval2D, r::_IA2DRel{R1,R2} where {R1<:_IA2DRelMaximizer,R2<:_IA2DRelSingleVal}, X::Integer, Y::Integer) =
	enumAccRepr2D(test_operator, w, r.x, r.y, X, Y, _ReprMin)
enumAccRepr(test_operator::_TestOpLes, w::Interval2D, r::_IA2DRel{R1,R2} where {R1<:_IA2DRelMinimizer,R2<:_IA2DRelSingleVal}, X::Integer, Y::Integer) =
	enumAccRepr2D(test_operator, w, r.x, r.y, X, Y, _ReprMax)
enumAccRepr(test_operator::_TestOpLes, w::Interval2D, r::_IA2DRel{R1,R2} where {R1<:_IA2DRelSingleVal,R2<:_IA2DRelMaximizer}, X::Integer, Y::Integer) =
	enumAccRepr2D(test_operator, w, r.x, r.y, X, Y, _ReprMin)
enumAccRepr(test_operator::_TestOpLes, w::Interval2D, r::_IA2DRel{R1,R2} where {R1<:_IA2DRelSingleVal,R2<:_IA2DRelMinimizer}, X::Integer, Y::Integer) =
	enumAccRepr2D(test_operator, w, r.x, r.y, X, Y, _ReprMax)

# The last two cases are difficult to express with enumAccRepr, better do it at WExtremaModal instead

# TODO create a dedicated min/max combination representation?
yieldMinMaxCombinations(test_operator::_TestOpGeq, productRepr::_ReprTreatment, channel::MatricialChannel{T,2}, dims::Integer) where {T} = begin
	if productRepr == _ReprNone{Interval2D}()
		return typemin(T),typemax(T)
	end
	vals = readWorld(productRepr.w, channel)
	# TODO try: maximum(mapslices(minimum, vals, dims=1)),minimum(mapslices(maximum, vals, dims=1))
	extr = vec(mapslices(extrema, vals, dims=dims))
	# println(extr)
	maxExtrema(extr)
end

yieldMinMaxCombination(test_operator::_TestOpGeq, productRepr::_ReprTreatment, channel::MatricialChannel{T,2}, dims::Integer) where {T} = begin
	if productRepr == _ReprNone{Interval2D}()
		return typemin(T)
	end
	vals = readWorld(productRepr.w, channel)
	maximum(mapslices(minimum, vals, dims=dims))
end

yieldMinMaxCombination(test_operator::_TestOpLes, productRepr::_ReprTreatment, channel::MatricialChannel{T,2}, dims::Integer) where {T} = begin
	if productRepr == _ReprNone{Interval2D}()
		return typemax(T)
	end
	vals = readWorld(productRepr.w, channel)
	minimum(mapslices(maximum, vals, dims=dims))
end

WExtremaModal(test_operator::_TestOpGeq, w::Interval2D, r::_IA2DRel{R1,R2} where {R1<:_IA2DRelMinimizer,R2<:_IA2DRelMaximizer}, channel::MatricialChannel{T,2}) where {T} = begin
	yieldMinMaxCombinations(test_operator, enumAccRepr2D(test_operator, w, r.x, r.y, size(channel)..., _ReprFake), channel, 1)
end
WExtremeModal(test_operator::Union{_TestOpGeq,_TestOpLes}, w::Interval2D, r::_IA2DRel{R1,R2} where {R1<:_IA2DRelMinimizer,R2<:_IA2DRelMaximizer}, channel::MatricialChannel{T,2}) where {T} = begin
	yieldMinMaxCombination(test_operator, enumAccRepr2D(test_operator, w, r.x, r.y, size(channel)..., _ReprFake), channel, 1)
end
WExtremaModal(test_operator::_TestOpGeq, w::Interval2D, r::_IA2DRel{R1,R2} where {R1<:_IA2DRelMaximizer,R2<:_IA2DRelMinimizer}, channel::MatricialChannel{T,2}) where {T} = begin
	yieldMinMaxCombinations(test_operator, enumAccRepr2D(test_operator, w, r.x, r.y, size(channel)..., _ReprFake), channel, 2)
end
WExtremeModal(test_operator::Union{_TestOpGeq,_TestOpLes}, w::Interval2D, r::_IA2DRel{R1,R2} where {R1<:_IA2DRelMaximizer,R2<:_IA2DRelMinimizer}, channel::MatricialChannel{T,2}) where {T} = begin
	yieldMinMaxCombination(test_operator, enumAccRepr2D(test_operator, w, r.x, r.y, size(channel)..., _ReprFake), channel, 2)
end

# TODO: per _TestOpLes gli operatori si invertono

################################################################################
# END IA2D relations
################################################################################

################################################################################
# BEGIN Topological relations
################################################################################

# Topological relations
abstract type _TopoRel <: AbstractRelation end
struct _Topo_DC     <: _TopoRel end; const Topo_DC     = _Topo_DC();     # Disconnected
struct _Topo_EC     <: _TopoRel end; const Topo_EC     = _Topo_EC();     # Externally connected
struct _Topo_PO     <: _TopoRel end; const Topo_PO     = _Topo_PO();     # Partially overlapping
struct _Topo_TPP    <: _TopoRel end; const Topo_TPP    = _Topo_TPP();    # Tangential proper part
struct _Topo_TPPi   <: _TopoRel end; const Topo_TPPi   = _Topo_TPPi();   # Tangential proper part inverse
struct _Topo_NTPP   <: _TopoRel end; const Topo_NTPP   = _Topo_NTPP();   # Non-tangential proper part
struct _Topo_NTPPi  <: _TopoRel end; const Topo_NTPPi  = _Topo_NTPPi();  # Non-tangential proper part inverse

display_rel_short(::_Topo_DC)    = "DC"
display_rel_short(::_Topo_EC)    = "EC"
display_rel_short(::_Topo_PO)    = "PO"
display_rel_short(::_Topo_TPP)   = "TPP"
display_rel_short(::_Topo_TPPi)  = "T̅P̅P̅" # T̄P̄P̄
display_rel_short(::_Topo_NTPP)  = "NTPP"
display_rel_short(::_Topo_NTPPi) = "N̅T̅P̅P̅" # N̄T̄P̄P̄

const TopoRelations = [Topo_DC, Topo_EC, Topo_PO, Topo_TPP, Topo_TPPi, Topo_NTPP, Topo_NTPPi]

topo2IARelations(::_Topo_DC)     = [IA_L,  IA_Li]
topo2IARelations(::_Topo_EC)     = [IA_A,  IA_Ai]
topo2IARelations(::_Topo_PO)     = [IA_O,  IA_Oi]
topo2IARelations(::_Topo_TPP)    = [IA_B,  IA_E]
topo2IARelations(::_Topo_TPPi)   = [IA_Bi, IA_Ei]
topo2IARelations(::_Topo_NTPP)   = [IA_D]
topo2IARelations(::_Topo_NTPPi)  = [IA_Di]

# Enumerate accessible worlds from a single world
enumAccBare(w::Interval, ::_Topo_DC,    X::Integer) = Iterators.flatten((enumAccBare(w, IA_L,  X), enumAccBare(w, IA_Li, X)))
enumAccBare(w::Interval, ::_Topo_EC,    X::Integer) = Iterators.flatten((enumAccBare(w, IA_A,  X), enumAccBare(w, IA_Ai, X)))
enumAccBare(w::Interval, ::_Topo_PO,    X::Integer) = Iterators.flatten((enumAccBare(w, IA_O,  X), enumAccBare(w, IA_Oi, X)))
enumAccBare(w::Interval, ::_Topo_TPP,   X::Integer) = Iterators.flatten((enumAccBare(w, IA_B,  X), enumAccBare(w, IA_E,  X)))
enumAccBare(w::Interval, ::_Topo_TPPi,  X::Integer) = Iterators.flatten((enumAccBare(w, IA_Bi, X), enumAccBare(w, IA_Ei, X)))
enumAccBare(w::Interval, ::_Topo_NTPP,  X::Integer) = enumAccBare(w, IA_D, X)
enumAccBare(w::Interval, ::_Topo_NTPPi, X::Integer) = enumAccBare(w, IA_Di, X)

# TODO write these compactly!!! Also softened operators can be written in compact form
# WExtremaModal(test_operator::_TestOpGeq, w::Interval, r::Union{_Topo_DC,_Topo_EC,_Topo_PO,_Topo_TPP,_Topo_TPPi}, channel::MatricialChannel{T,1}) where {T} = begin
# 	repr1, repr2 = map((r_IA)->(enumAccRepr(w, r_IA,  X)), topo2IARelations(r))
# 	extr = yieldReprs(test_operator, repr1, channel), yieldReprs(test_operator, repr2, channel)
# 	maxExtrema(extr)
# end
# WExtremeModal(test_operator::_TestOpGeq, w::Interval, r::Union{_Topo_DC,_Topo_EC,_Topo_PO,_Topo_TPP,_Topo_TPPi}, channel::MatricialChannel{T,1}) where {T} = begin
# 	repr1, repr2 = map((r_IA)->(enumAccRepr(w, r_IA,  X)), topo2IARelations(r))
# 	max(yieldRepr(test_operator, repr1, channel), yieldRepr(test_operator, repr2, channel))
# end
# WExtremeModal(test_operator::_TestOpLes, w::Interval, r::Union{_Topo_DC,_Topo_EC,_Topo_PO,_Topo_TPP,_Topo_TPPi}, channel::MatricialChannel{T,1}) where {T} = begin
# 	repr1, repr2 = map((r_IA)->(enumAccRepr(w, r_IA,  X)), topo2IARelations(r))
# 	min(yieldRepr(test_operator, repr1, channel), yieldRepr(test_operator, repr2, channel))
# end

WExtremaModal(test_operator::_TestOpGeq, w::Interval, r::_Topo_DC, channel::MatricialChannel{T,1}) where {T} = begin
	repr1 = enumAccRepr(test_operator, w, IA_L, length(channel))
	repr2 = enumAccRepr(test_operator, w, IA_Li, length(channel))
	extr = yieldReprs(test_operator, repr1, channel), yieldReprs(test_operator, repr2, channel)
	maxExtrema(extr)
end
WExtremeModal(test_operator::_TestOpGeq, w::Interval, r::_Topo_DC, channel::MatricialChannel{T,1}) where {T} = begin
	repr1 = enumAccRepr(test_operator, w, IA_L, length(channel))
	repr2 = enumAccRepr(test_operator, w, IA_Li, length(channel))
	max(yieldRepr(test_operator, repr1, channel), yieldRepr(test_operator, repr2, channel))
end
WExtremeModal(test_operator::_TestOpLes, w::Interval, r::_Topo_DC, channel::MatricialChannel{T,1}) where {T} = begin
	repr1 = enumAccRepr(test_operator, w, IA_L, length(channel))
	repr2 = enumAccRepr(test_operator, w, IA_Li, length(channel))
	min(yieldRepr(test_operator, repr1, channel), yieldRepr(test_operator, repr2, channel))
end

WExtremaModal(test_operator::_TestOpGeq, w::Interval, r::_Topo_EC, channel::MatricialChannel{T,1}) where {T} = begin
	repr1 = enumAccRepr(test_operator, w, IA_A, length(channel))
	repr2 = enumAccRepr(test_operator, w, IA_Ai, length(channel))
	extr = yieldReprs(test_operator, repr1, channel), yieldReprs(test_operator, repr2, channel)
	maxExtrema(extr)
end
WExtremeModal(test_operator::_TestOpGeq, w::Interval, r::_Topo_EC, channel::MatricialChannel{T,1}) where {T} = begin
	repr1 = enumAccRepr(test_operator, w, IA_A, length(channel))
	repr2 = enumAccRepr(test_operator, w, IA_Ai, length(channel))
	max(yieldRepr(test_operator, repr1, channel), yieldRepr(test_operator, repr2, channel))
end
WExtremeModal(test_operator::_TestOpLes, w::Interval, r::_Topo_EC, channel::MatricialChannel{T,1}) where {T} = begin
	repr1 = enumAccRepr(test_operator, w, IA_A, length(channel))
	repr2 = enumAccRepr(test_operator, w, IA_Ai, length(channel))
	min(yieldRepr(test_operator, repr1, channel), yieldRepr(test_operator, repr2, channel))
end

WExtremaModal(test_operator::_TestOpGeq, w::Interval, r::_Topo_PO, channel::MatricialChannel{T,1}) where {T} = begin
	repr1 = enumAccRepr(test_operator, w, IA_O, length(channel))
	repr2 = enumAccRepr(test_operator, w, IA_Oi, length(channel))
	extr = yieldReprs(test_operator, repr1, channel), yieldReprs(test_operator, repr2, channel)
	maxExtrema(extr)
end
WExtremeModal(test_operator::_TestOpGeq, w::Interval, r::_Topo_PO, channel::MatricialChannel{T,1}) where {T} = begin
	repr1 = enumAccRepr(test_operator, w, IA_O, length(channel))
	repr2 = enumAccRepr(test_operator, w, IA_Oi, length(channel))
	max(yieldRepr(test_operator, repr1, channel), yieldRepr(test_operator, repr2, channel))
end
WExtremeModal(test_operator::_TestOpLes, w::Interval, r::_Topo_PO, channel::MatricialChannel{T,1}) where {T} = begin
	repr1 = enumAccRepr(test_operator, w, IA_O, length(channel))
	repr2 = enumAccRepr(test_operator, w, IA_Oi, length(channel))
	min(yieldRepr(test_operator, repr1, channel), yieldRepr(test_operator, repr2, channel))
end

WExtremaModal(test_operator::_TestOpGeq, w::Interval, r::_Topo_TPP, channel::MatricialChannel{T,1}) where {T} = begin
	repr1 = enumAccRepr(test_operator, w, IA_B, length(channel))
	repr2 = enumAccRepr(test_operator, w, IA_E, length(channel))
	extr = yieldReprs(test_operator, repr1, channel), yieldReprs(test_operator, repr2, channel)
	maxExtrema(extr)
end
WExtremeModal(test_operator::_TestOpGeq, w::Interval, r::_Topo_TPP, channel::MatricialChannel{T,1}) where {T} = begin
	repr1 = enumAccRepr(test_operator, w, IA_B, length(channel))
	repr2 = enumAccRepr(test_operator, w, IA_E, length(channel))
	max(yieldRepr(test_operator, repr1, channel), yieldRepr(test_operator, repr2, channel))
end
WExtremeModal(test_operator::_TestOpLes, w::Interval, r::_Topo_TPP, channel::MatricialChannel{T,1}) where {T} = begin
	repr1 = enumAccRepr(test_operator, w, IA_B, length(channel))
	repr2 = enumAccRepr(test_operator, w, IA_E, length(channel))
	min(yieldRepr(test_operator, repr1, channel), yieldRepr(test_operator, repr2, channel))
end

WExtremaModal(test_operator::_TestOpGeq, w::Interval, r::_Topo_TPPi, channel::MatricialChannel{T,1}) where {T} = begin
	repr1 = enumAccRepr(test_operator, w, IA_Bi, length(channel))
	repr2 = enumAccRepr(test_operator, w, IA_Ei, length(channel))
	extr = yieldReprs(test_operator, repr1, channel), yieldReprs(test_operator, repr2, channel)
	maxExtrema(extr)
end
WExtremeModal(test_operator::_TestOpGeq, w::Interval, r::_Topo_TPPi, channel::MatricialChannel{T,1}) where {T} = begin
	repr1 = enumAccRepr(test_operator, w, IA_Bi, length(channel))
	repr2 = enumAccRepr(test_operator, w, IA_Ei, length(channel))
	max(yieldRepr(test_operator, repr1, channel), yieldRepr(test_operator, repr2, channel))
end
WExtremeModal(test_operator::_TestOpLes, w::Interval, r::_Topo_TPPi, channel::MatricialChannel{T,1}) where {T} = begin
	repr1 = enumAccRepr(test_operator, w, IA_Bi, length(channel))
	repr2 = enumAccRepr(test_operator, w, IA_Ei, length(channel))
	min(yieldRepr(test_operator, repr1, channel), yieldRepr(test_operator, repr2, channel))
end

enumAccRepr(test_operator::TestOperator, w::Interval, ::_Topo_NTPP,  X::Integer) = enumAccRepr(test_operator, w, IA_D, X)
enumAccRepr(test_operator::TestOperator, w::Interval, ::_Topo_NTPPi, X::Integer) = enumAccRepr(test_operator, w, IA_Di, X)


# More efficient implementations for edge cases
# ?

# TODO try building Interval's in these, because they are only bare with respect to Interval2D
# Enumerate accessible worlds from a single world
# TODO try inverting these two?
enumAccBare(w::Interval2D, ::_Topo_DC,    X::Integer, Y::Integer) =
	IterTools.distinct(
		Iterators.flatten((
			Iterators.product(enumAccBare(w.x, Topo_DC,    X), enumAccBare2(w.y, RelationAll,Y)),
			Iterators.product(enumAccBare2(w.x, RelationAll,X), enumAccBare(w.y, Topo_DC,    Y)),
			# TODO try avoiding the distinct, replacing the second line (RelationAll,enumAccBare) with 7 combinations of RelationAll with Topo_EC, Topo_PO, Topo_TPP, Topo_TPPi, Topo_NTPP, Topo_NTPPi
		))
	)
enumAccBare(w::Interval2D, ::_Topo_EC,    X::Integer, Y::Integer) =
	Iterators.flatten((
		Iterators.product(enumAccBare(w.x, Topo_EC,    X), enumAccBare(w.y, Topo_EC,    Y)),
		#
		Iterators.product(enumAccBare(w.x, Topo_EC,    X), enumAccBare(w.y, Topo_PO,    Y)),
		Iterators.product(enumAccBare(w.x, Topo_EC,    X), enumAccBare(w.y, Topo_TPP,   Y)),
		Iterators.product(enumAccBare(w.x, Topo_EC,    X), enumAccBare(w.y, Topo_TPPi,  Y)),
		Iterators.product(enumAccBare(w.x, Topo_EC,    X), enumAccBare(w.y, Topo_NTPP,  Y)),
		Iterators.product(enumAccBare(w.x, Topo_EC,    X), enumAccBare(w.y, Topo_NTPPi, Y)),
		Iterators.product(enumAccBare(w.x, Topo_EC,    X), enumAccBare(w.y, RelationId, Y)),
		#
		Iterators.product(enumAccBare(w.x, Topo_PO,    X), enumAccBare(w.y, Topo_EC,    Y)),
		Iterators.product(enumAccBare(w.x, Topo_TPP,   X), enumAccBare(w.y, Topo_EC,    Y)),
		Iterators.product(enumAccBare(w.x, Topo_TPPi,  X), enumAccBare(w.y, Topo_EC,    Y)),
		Iterators.product(enumAccBare(w.x, Topo_NTPP,  X), enumAccBare(w.y, Topo_EC,    Y)),
		Iterators.product(enumAccBare(w.x, Topo_NTPPi, X), enumAccBare(w.y, Topo_EC,    Y)),
		Iterators.product(enumAccBare(w.x, RelationId, X), enumAccBare(w.y, Topo_EC,    Y)),
	))
enumAccBare(w::Interval2D, ::_Topo_PO,    X::Integer, Y::Integer) =
	Iterators.flatten((
		Iterators.product(enumAccBare(w.x, Topo_PO,    X), enumAccBare(w.y, Topo_PO,    Y)),
		#
		Iterators.product(enumAccBare(w.x, Topo_TPP,   X), enumAccBare(w.y, Topo_PO,    Y)),
		Iterators.product(enumAccBare(w.x, Topo_TPPi,  X), enumAccBare(w.y, Topo_PO,    Y)),
		Iterators.product(enumAccBare(w.x, Topo_NTPP,  X), enumAccBare(w.y, Topo_PO,    Y)),
		Iterators.product(enumAccBare(w.x, Topo_NTPPi, X), enumAccBare(w.y, Topo_PO,    Y)),
		Iterators.product(enumAccBare(w.x, RelationId, X), enumAccBare(w.y, Topo_PO,    Y)),
		#
		Iterators.product(enumAccBare(w.x, Topo_PO,    X), enumAccBare(w.y, Topo_TPP,   Y)),
		Iterators.product(enumAccBare(w.x, Topo_PO,    X), enumAccBare(w.y, Topo_TPPi,  Y)),
		Iterators.product(enumAccBare(w.x, Topo_PO,    X), enumAccBare(w.y, Topo_NTPP,  Y)),
		Iterators.product(enumAccBare(w.x, Topo_PO,    X), enumAccBare(w.y, Topo_NTPPi, Y)),
		Iterators.product(enumAccBare(w.x, Topo_PO,    X), enumAccBare(w.y, RelationId, Y)),
		#
		Iterators.product(enumAccBare(w.x, Topo_TPP,   X), enumAccBare(w.y, Topo_TPPi,  Y)),
		Iterators.product(enumAccBare(w.x, Topo_TPPi,  X), enumAccBare(w.y, Topo_TPP,   Y)),
		#
		Iterators.product(enumAccBare(w.x, Topo_TPPi,  X), enumAccBare(w.y, Topo_NTPP,  Y)),
		Iterators.product(enumAccBare(w.x, Topo_NTPP,  X), enumAccBare(w.y, Topo_TPPi,  Y)),
		#
		Iterators.product(enumAccBare(w.x, Topo_TPP,   X), enumAccBare(w.y, Topo_NTPPi, Y)),
		Iterators.product(enumAccBare(w.x, Topo_NTPP,  X), enumAccBare(w.y, Topo_NTPPi, Y)),
		Iterators.product(enumAccBare(w.x, Topo_NTPPi, X), enumAccBare(w.y, Topo_TPP,   Y)),
		Iterators.product(enumAccBare(w.x, Topo_NTPPi, X), enumAccBare(w.y, Topo_NTPP,  Y)),
	))
enumAccBare(w::Interval2D, ::_Topo_TPP,   X::Integer, Y::Integer) =
	Iterators.flatten((
		Iterators.product(enumAccBare(w.x, Topo_TPP,   X), enumAccBare(w.y, Topo_TPP,   Y)),
		#
		Iterators.product(enumAccBare(w.x, Topo_TPP,   X), enumAccBare(w.y, Topo_NTPP,  Y)),
		Iterators.product(enumAccBare(w.x, Topo_NTPP,  X), enumAccBare(w.y, Topo_TPP,   Y)),
		#
		Iterators.product(enumAccBare(w.x, Topo_TPP,   X), enumAccBare(w.y, RelationId, Y)),
		Iterators.product(enumAccBare(w.x, RelationId, X), enumAccBare(w.y, Topo_TPP,   Y)),
		#
		Iterators.product(enumAccBare(w.x, Topo_NTPP,  X), enumAccBare(w.y, RelationId, Y)),
		Iterators.product(enumAccBare(w.x, RelationId, X), enumAccBare(w.y, Topo_NTPP,  Y)),
	))

enumAccBare(w::Interval2D, ::_Topo_TPPi,  X::Integer, Y::Integer) =
	Iterators.flatten((
		Iterators.product(enumAccBare(w.x, Topo_TPPi,  X), enumAccBare(w.y, Topo_TPPi,  Y)),
		#
		Iterators.product(enumAccBare(w.x, Topo_TPPi,  X), enumAccBare(w.y, Topo_NTPPi, Y)),
		Iterators.product(enumAccBare(w.x, Topo_NTPPi, X), enumAccBare(w.y, Topo_TPPi,  Y)),
		#
		Iterators.product(enumAccBare(w.x, Topo_TPPi,  X), enumAccBare(w.y, RelationId, Y)),
		Iterators.product(enumAccBare(w.x, RelationId, X), enumAccBare(w.y, Topo_TPPi,  Y)),
		#
		Iterators.product(enumAccBare(w.x, Topo_NTPPi, X), enumAccBare(w.y, RelationId, Y)),
		Iterators.product(enumAccBare(w.x, RelationId, X), enumAccBare(w.y, Topo_NTPPi, Y)),
	))

enumAccBare(w::Interval2D, ::_Topo_NTPP,  X::Integer, Y::Integer) =
	# Iterators.flatten((
		Iterators.product(enumAccBare(w.x, Topo_NTPP,  X), enumAccBare(w.y, Topo_NTPP,  Y))
		# , ))
enumAccBare(w::Interval2D, ::_Topo_NTPPi, X::Integer, Y::Integer) =
	# Iterators.flatten((
		Iterators.product(enumAccBare(w.x, Topo_NTPPi, X), enumAccBare(w.y, Topo_NTPPi, Y))
	# , ))


# Virtual relation used for computing Topo_DC on Interval2D
struct _Virtual_Enlarge <: AbstractRelation end; const Virtual_Enlarge = _Virtual_Enlarge();     # Virtual_Enlarge
enlargeInterval(w::Interval, X::Integer) = Interval(max(1,w.x-1),min(w.y+1,X+1))

enumAccRepr(test_operator::_TestOpGeq, w::Interval, ::_Virtual_Enlarge,  X::Integer) = _ReprMin(enlargeInterval(w,X))
enumAccRepr(test_operator::_TestOpLes, w::Interval, ::_Virtual_Enlarge,  X::Integer) = _ReprMax(enlargeInterval(w,X))


# Topo2D2Topo1D(::_Topo_DC) = [
# 															(RelationAll , Topo_DC),
# 															# TODO many many others but for now let's just say...
# 															(Topo_DC     , Virtual_Enlarge),
# ]
Topo2D2Topo1D(::_Topo_EC) = [
															(Topo_EC     , Topo_EC),
															#
															(Topo_PO     , Topo_EC),
															(Topo_TPP    , Topo_EC),
															(Topo_TPPi   , Topo_EC),
															(Topo_NTPP   , Topo_EC),
															(Topo_NTPPi  , Topo_EC),
															(RelationId  , Topo_EC),
															#
															(Topo_EC     , Topo_PO),
															(Topo_EC     , Topo_TPP),
															(Topo_EC     , Topo_TPPi),
															(Topo_EC     , Topo_NTPP),
															(Topo_EC     , Topo_NTPPi),
															(Topo_EC     , RelationId),
]
Topo2D2Topo1D(::_Topo_PO) = [
															(Topo_PO     , Topo_PO),
															#
															(Topo_PO     , Topo_TPP),
															(Topo_PO     , Topo_TPPi),
															(Topo_PO     , Topo_NTPP),
															(Topo_PO     , Topo_NTPPi),
															(Topo_PO     , RelationId),
															#
															(Topo_TPP    , Topo_PO),
															(Topo_TPPi   , Topo_PO),
															(Topo_NTPP   , Topo_PO),
															(Topo_NTPPi  , Topo_PO),
															(RelationId  , Topo_PO),
															#
															(Topo_TPPi   , Topo_TPP),
															(Topo_TPP    , Topo_TPPi),
															#
															(Topo_NTPP   , Topo_TPPi),
															(Topo_TPPi   , Topo_NTPP),
															#
															(Topo_NTPPi  , Topo_TPP),
															(Topo_NTPPi  , Topo_NTPP),
															(Topo_TPP    , Topo_NTPPi),
															(Topo_NTPP   , Topo_NTPPi),
]
Topo2D2Topo1D(::_Topo_TPP) = [
															(Topo_TPP    , Topo_TPP),   
															#
															(Topo_NTPP   , Topo_TPP),   
															(Topo_TPP    , Topo_NTPP),  
															#
															(RelationId  , Topo_TPP),   
															(Topo_TPP    , RelationId), 
															#
															(RelationId  , Topo_NTPP),  
															(Topo_NTPP   , RelationId), 
]
Topo2D2Topo1D(::_Topo_TPPi) = [
															(Topo_TPPi   , Topo_TPPi),
															#
															(Topo_NTPPi  , Topo_TPPi),
															(Topo_TPPi   , Topo_NTPPi),
															#
															(RelationId  , Topo_TPPi),
															(Topo_TPPi   , RelationId),
															#
															(RelationId  , Topo_NTPPi),
															(Topo_NTPPi  , RelationId),
]
Topo2D2Topo1D(::_Topo_NTPP) = [
															(Topo_NTPP   , Topo_NTPP),
]
Topo2D2Topo1D(::_Topo_NTPPi) = [
															(Topo_NTPPi  , Topo_NTPPi),
]


WExtremaModal(test_operator::_TestOpGeq, w::Interval2D, r::_Topo_DC, channel::MatricialChannel{T,2}) where {T} = begin
	reprx1 = enumAccRepr2D(test_operator, w, RelationAll, IA_L,         size(channel)..., _ReprMax)
	reprx2 = enumAccRepr2D(test_operator, w, RelationAll, IA_Li,        size(channel)..., _ReprMax)
	repry1 = enumAccRepr2D(test_operator, w, IA_L,     Virtual_Enlarge, size(channel)..., _ReprMax)
	repry2 = enumAccRepr2D(test_operator, w, IA_Li,    Virtual_Enlarge, size(channel)..., _ReprMax)
	extr = yieldReprs(test_operator, reprx1, channel),
				 yieldReprs(test_operator, reprx2, channel),
				 yieldReprs(test_operator, repry1, channel),
				 yieldReprs(test_operator, repry2, channel)
	maxExtrema(extr)
end
WExtremeModal(test_operator::_TestOpGeq, w::Interval2D, r::_Topo_DC, channel::MatricialChannel{T,2}) where {T} = begin
	# reprx1 = enumAccRepr2D(test_operator, w, IA_L,         RelationAll, size(channel)..., _ReprMax)
	# reprx2 = enumAccRepr2D(test_operator, w, IA_Li,        RelationAll, size(channel)..., _ReprMax)
	# repry1 = enumAccRepr2D(test_operator, w, RelationAll,  IA_L,        size(channel)..., _ReprMax)
	# repry2 = enumAccRepr2D(test_operator, w, RelationAll,  IA_Li,       size(channel)..., _ReprMax)
	reprx1 = enumAccRepr2D(test_operator, w, RelationAll, IA_L,         size(channel)..., _ReprMax)
	reprx2 = enumAccRepr2D(test_operator, w, RelationAll, IA_Li,        size(channel)..., _ReprMax)
	repry1 = enumAccRepr2D(test_operator, w, IA_L,     Virtual_Enlarge, size(channel)..., _ReprMax)
	repry2 = enumAccRepr2D(test_operator, w, IA_Li,    Virtual_Enlarge, size(channel)..., _ReprMax)
	# if channel == [819 958 594; 749 665 383; 991 493 572] && w.x.x==1 && w.x.y==2 && w.y.x==1 && w.y.y==3
	# 	println(max(yieldRepr(test_operator, reprx1, channel),
	# 			 yieldRepr(test_operator, reprx2, channel),
	# 			 yieldRepr(test_operator, repry1, channel),
	# 			 yieldRepr(test_operator, repry2, channel)))
	# 	readline()
	# end
	max(yieldRepr(test_operator, reprx1, channel),
				 yieldRepr(test_operator, reprx2, channel),
				 yieldRepr(test_operator, repry2, channel),
				 yieldRepr(test_operator, repry2, channel))
end
WExtremeModal(test_operator::_TestOpLes, w::Interval2D, r::_Topo_DC, channel::MatricialChannel{T,2}) where {T} = begin
	reprx1 = enumAccRepr2D(test_operator, w, RelationAll, IA_L,         size(channel)..., _ReprMin)
	reprx2 = enumAccRepr2D(test_operator, w, RelationAll, IA_Li,        size(channel)..., _ReprMin)
	repry1 = enumAccRepr2D(test_operator, w, IA_L,     Virtual_Enlarge, size(channel)..., _ReprMin)
	repry2 = enumAccRepr2D(test_operator, w, IA_Li,    Virtual_Enlarge, size(channel)..., _ReprMin)
	min(yieldRepr(test_operator, reprx1, channel),
				 yieldRepr(test_operator, reprx2, channel),
				 yieldRepr(test_operator, repry1, channel),
				 yieldRepr(test_operator, repry2, channel))
end

# EC: Just optimize the values on the outer boundary
WExtremaModal(test_operator::_TestOpGeq, w::Interval2D, r::_Topo_EC, channel::MatricialChannel{T,2}) where {T} = begin
	X,Y = size(channel)
	reprs = [
		((w.x.x-1 >= 1)   ? [Interval2D(Interval(w.x.x-1,w.x.x),enlargeInterval(w.y,Y))] : Interval2D[])...,
		((w.x.y+1 <= X+1) ? [Interval2D(Interval(w.x.y,w.x.y+1),enlargeInterval(w.y,Y))] : Interval2D[])...,
		((w.y.x-1 >= 1)   ? [Interval2D(enlargeInterval(w.x,X),Interval(w.y.x-1,w.y.x))] : Interval2D[])...,
		((w.y.y+1 <= Y+1) ? [Interval2D(enlargeInterval(w.x,X),Interval(w.y.y,w.y.y+1))] : Interval2D[])...,
	]
	extr = map(w->yieldReprs(test_operator, _ReprMax(w), channel), reprs)
	maxExtrema(extr)
end
WExtremeModal(test_operator::_TestOpGeq, w::Interval2D, r::_Topo_EC, channel::MatricialChannel{T,2}) where {T} = begin
	X,Y = size(channel)
	reprs = [
		((w.x.x-1 >= 1)   ? [Interval2D(Interval(w.x.x-1,w.x.x),enlargeInterval(w.y,Y))] : Interval2D[])...,
		((w.x.y+1 <= X+1) ? [Interval2D(Interval(w.x.y,w.x.y+1),enlargeInterval(w.y,Y))] : Interval2D[])...,
		((w.y.x-1 >= 1)   ? [Interval2D(enlargeInterval(w.x,X),Interval(w.y.x-1,w.y.x))] : Interval2D[])...,
		((w.y.y+1 <= Y+1) ? [Interval2D(enlargeInterval(w.x,X),Interval(w.y.y,w.y.y+1))] : Interval2D[])...,
	]
	extr = map(w->yieldRepr(test_operator, _ReprMax(w), channel), reprs)
	maximum([extr..., typemin(T)])
end
WExtremeModal(test_operator::_TestOpLes, w::Interval2D, r::_Topo_EC, channel::MatricialChannel{T,2}) where {T} = begin
	X,Y = size(channel)
	reprs = [
		((w.x.x-1 >= 1)   ? [Interval2D(Interval(w.x.x-1,w.x.x),enlargeInterval(w.y,Y))] : Interval2D[])...,
		((w.x.y+1 <= X+1) ? [Interval2D(Interval(w.x.y,w.x.y+1),enlargeInterval(w.y,Y))] : Interval2D[])...,
		((w.y.x-1 >= 1)   ? [Interval2D(enlargeInterval(w.x,X),Interval(w.y.x-1,w.y.x))] : Interval2D[])...,
		((w.y.y+1 <= Y+1) ? [Interval2D(enlargeInterval(w.x,X),Interval(w.y.y,w.y.y+1))] : Interval2D[])...,
	]
	extr = map(w->yieldRepr(test_operator, _ReprMin(w), channel), reprs)
	minimum([extr..., typemax(T)])
end

# PO: For each pair crossing the border, perform a minimization step and then a maximization step
WExtremaModal(test_operator::_TestOpGeq, w::Interval2D, r::_Topo_PO, channel::MatricialChannel{T,2}) where {T} = begin
	# if true &&
	# 	# (channel == [1620 1408 1343; 1724 1398 1252; 1177 1703 1367] && w.x.x==1 && w.x.y==3 && w.y.x==3 && w.y.y==4) ||
	# 	# (channel == [412 489 559 619 784; 795 771 1317 854 1256; 971 874 878 1278 560] && w.x.x==1 && w.x.y==3 && w.y.x==3 && w.y.y==4)
	# 	(channel == [2405 2205 1898 1620 1383; 1922 1555 1383 1393 1492; 1382 1340 1434 1640 1704] && w.x.x==1 && w.x.y==3 && w.y.x==3 && w.y.y==4)
		
	# 	x_singleton = ! (w.x.x < w.x.y-1)
	# 	y_singleton = ! (w.y.x < w.y.y-1)
	# 	if x_singleton && y_singleton
	# 		println(typemin(T),typemax(T))
	# 	else
	# 		rx1,rx2 = x_singleton ? (IA_Bi,IA_Ei) : (IA_O,IA_Oi)
	# 		ry1,ry2 = y_singleton ? (IA_Bi,IA_Ei) : (IA_O,IA_Oi)
	# 		println(rx1)
	# 		println(rx2)
	# 		println(ry1)
	# 		println(ry2)
	# 		# reprx1 = enumAccRepr2D(test_operator, w, rx1,    RelationId, size(channel)..., _ReprMin)
	# 		# reprx2 = enumAccRepr2D(test_operator, w, rx2,    RelationId, size(channel)..., _ReprMin)
	# 		# repry1 = enumAccRepr2D(test_operator, w, RelationId, ry1,        size(channel)..., _ReprMin)
	# 		# repry2 = enumAccRepr2D(test_operator, w, RelationId, ry2,        size(channel)..., _ReprMin)
	# 		# println(reprx1)
	# 		# println(reprx2)
	# 		# println(repry1)
	# 		# println(repry2)

	# 		println(
	# 		yieldMinMaxCombinations(test_operator, enumAccRepr2D(test_operator, w, RelationId, ry1,    size(channel)..., _ReprFake), channel, 2)
	# 		)
	# 		println(
	# 		yieldMinMaxCombinations(test_operator, enumAccRepr2D(test_operator, w, RelationId, ry2,    size(channel)..., _ReprFake), channel, 2)
	# 		)
	# 		println(
	# 		yieldMinMaxCombinations(test_operator, enumAccRepr2D(test_operator, w, rx1,    RelationId, size(channel)..., _ReprFake), channel, 1)
	# 		)
	# 		println(
	# 		yieldMinMaxCombinations(test_operator, enumAccRepr2D(test_operator, w, rx2,    RelationId, size(channel)..., _ReprFake), channel, 1)
	# 		)
	# 		println(
	# 		maxExtrema((
	# 			yieldMinMaxCombinations(test_operator, enumAccRepr2D(test_operator, w, RelationId, ry1,    size(channel)..., _ReprFake), channel, 2),
	# 			yieldMinMaxCombinations(test_operator, enumAccRepr2D(test_operator, w, RelationId, ry2,    size(channel)..., _ReprFake), channel, 2),
	# 			yieldMinMaxCombinations(test_operator, enumAccRepr2D(test_operator, w, rx1,    RelationId, size(channel)..., _ReprFake), channel, 1),
	# 			yieldMinMaxCombinations(test_operator, enumAccRepr2D(test_operator, w, rx2,    RelationId, size(channel)..., _ReprFake), channel, 1),
	# 		))
	# 		)

	# 		# println(WExtremaModal(test_operator, w, _IA2DRel(rx1        , RelationId), channel))
	# 		# println(WExtremaModal(test_operator, w, _IA2DRel(rx2        , RelationId), channel))
	# 		# println(WExtremaModal(test_operator, w, _IA2DRel(RelationId , ry1),        channel))
	# 		# println(WExtremaModal(test_operator, w, _IA2DRel(RelationId , ry2),        channel))
	# 		# println(maxExtrema((
	# 		# 	WExtremaModal(test_operator, w, _IA2DRel(rx1        , RelationId), channel),
	# 		# 	WExtremaModal(test_operator, w, _IA2DRel(rx2        , RelationId), channel),
	# 		# 	WExtremaModal(test_operator, w, _IA2DRel(RelationId , ry1),        channel),
	# 		# 	WExtremaModal(test_operator, w, _IA2DRel(RelationId , ry2),        channel),
	# 		# 	))
	# 		# )
	# 	end

	# 	readline()
	# end
	x_singleton = ! (w.x.x < w.x.y-1)
	y_singleton = ! (w.y.x < w.y.y-1)
	if x_singleton && y_singleton
		return typemin(T),typemax(T)
	end

	rx1,rx2 = x_singleton ? (IA_Bi,IA_Ei) : (IA_O,IA_Oi)
	ry1,ry2 = y_singleton ? (IA_Bi,IA_Ei) : (IA_O,IA_Oi)

	# reprx1 = enumAccRepr2D(test_operator, w, rx1,    RelationId, size(channel)..., _ReprFake)
	# reprx2 = enumAccRepr2D(test_operator, w, rx2,    RelationId, size(channel)..., _ReprFake)
	# repry1 = enumAccRepr2D(test_operator, w, RelationId, ry1,        size(channel)..., _ReprFake)
	# repry2 = enumAccRepr2D(test_operator, w, RelationId, ry2,        size(channel)..., _ReprFake)

	maxExtrema((
		yieldMinMaxCombinations(test_operator, enumAccRepr2D(test_operator, w, RelationId, ry1,    size(channel)..., _ReprFake), channel, 2),
		yieldMinMaxCombinations(test_operator, enumAccRepr2D(test_operator, w, RelationId, ry2,    size(channel)..., _ReprFake), channel, 2),
		yieldMinMaxCombinations(test_operator, enumAccRepr2D(test_operator, w, rx1,    RelationId, size(channel)..., _ReprFake), channel, 1),
		yieldMinMaxCombinations(test_operator, enumAccRepr2D(test_operator, w, rx2,    RelationId, size(channel)..., _ReprFake), channel, 1),
	))
end
WExtremeModal(test_operator::_TestOpGeq, w::Interval2D, r::_Topo_PO, channel::MatricialChannel{T,2}) where {T} = begin
	# if channel == [1620 1408 1343; 1724 1398 1252; 1177 1703 1367] && w.x.x==1 && w.x.y==3 && w.y.x==3 && w.y.y==4
	# 	println(! (w.x.x < w.x.y-1) && ! (w.y.x < w.y.y-1))
	# 	println(max(
	# 		WExtremaModal(test_operator, w, _IA2DRel(RelationId , IA_O),       channel),
	# 		WExtremaModal(test_operator, w, _IA2DRel(RelationId , IA_Oi),      channel),
	# 		WExtremaModal(test_operator, w, _IA2DRel(IA_Oi      , RelationId), channel),
	# 		WExtremaModal(test_operator, w, _IA2DRel(IA_O       , RelationId), channel),
	# 	))
	# 	readline()
	# end
	x_singleton = ! (w.x.x < w.x.y-1)
	y_singleton = ! (w.y.x < w.y.y-1)
	if x_singleton && y_singleton
		return typemin(T)
	end

	rx1,rx2 = x_singleton ? (IA_Bi,IA_Ei) : (IA_O,IA_Oi)
	ry1,ry2 = y_singleton ? (IA_Bi,IA_Ei) : (IA_O,IA_Oi)

	max((
		yieldMinMaxCombination(test_operator, enumAccRepr2D(test_operator, w, RelationId, ry1,    size(channel)..., _ReprFake), channel, 2),
		yieldMinMaxCombination(test_operator, enumAccRepr2D(test_operator, w, RelationId, ry2,    size(channel)..., _ReprFake), channel, 2),
		yieldMinMaxCombination(test_operator, enumAccRepr2D(test_operator, w, rx1,    RelationId, size(channel)..., _ReprFake), channel, 1),
		yieldMinMaxCombination(test_operator, enumAccRepr2D(test_operator, w, rx2,    RelationId, size(channel)..., _ReprFake), channel, 1),
	))
end
WExtremeModal(test_operator::_TestOpLes, w::Interval2D, r::_Topo_PO, channel::MatricialChannel{T,2}) where {T} = begin
	x_singleton = ! (w.x.x < w.x.y-1)
	y_singleton = ! (w.y.x < w.y.y-1)
	if x_singleton && y_singleton
		return typemax(T)
	end
	
	rx1,rx2 = x_singleton ? (IA_Bi,IA_Ei) : (IA_O,IA_Oi)
	ry1,ry2 = y_singleton ? (IA_Bi,IA_Ei) : (IA_O,IA_Oi)

	reprx1 = enumAccRepr2D(test_operator, w, rx1,    RelationAll, size(channel)..., _ReprMax)
	reprx2 = enumAccRepr2D(test_operator, w, rx2,    RelationAll, size(channel)..., _ReprMax)
	repry1 = enumAccRepr2D(test_operator, w, RelationAll, ry1,        size(channel)..., _ReprMax)
	repry2 = enumAccRepr2D(test_operator, w, RelationAll, ry2,        size(channel)..., _ReprMax)
	min((
		yieldMinMaxCombination(test_operator, enumAccRepr2D(test_operator, w, RelationId, ry1,    size(channel)..., _ReprFake), channel, 2),
		yieldMinMaxCombination(test_operator, enumAccRepr2D(test_operator, w, RelationId, ry2,    size(channel)..., _ReprFake), channel, 2),
		yieldMinMaxCombination(test_operator, enumAccRepr2D(test_operator, w, rx1,    RelationId, size(channel)..., _ReprFake), channel, 1),
		yieldMinMaxCombination(test_operator, enumAccRepr2D(test_operator, w, rx2,    RelationId, size(channel)..., _ReprFake), channel, 1),
	))
end

# TPP: Just optimize the values on the inner boundary
WExtremaModal(test_operator::_TestOpGeq, w::Interval2D, r::_Topo_TPP, channel::MatricialChannel{T,2}) where {T} = begin
	reprs = if (w.x.x < w.x.y-1) && (w.y.x < w.y.y-1)
			[Interval2D(Interval(w.x.x,w.x.x+1),w.y), Interval2D(Interval(w.x.y-1,w.x.y),w.y), Interval2D(w.x,Interval(w.y.x,w.y.x+1)), Interval2D(w.x,Interval(w.y.y-1,w.y.y))]
		elseif (w.x.x < w.x.y-1) || (w.y.x < w.y.y-1)
			[w]
		else Interval2D[]
	end
	extr = map(w->yieldReprs(test_operator, _ReprMax(w), channel), reprs)
	maxExtrema(extr)
end
WExtremeModal(test_operator::_TestOpGeq, w::Interval2D, r::_Topo_TPP, channel::MatricialChannel{T,2}) where {T} = begin
	reprs = if (w.x.x < w.x.y-1) && (w.y.x < w.y.y-1)
			[Interval2D(Interval(w.x.x,w.x.x+1),w.y), Interval2D(Interval(w.x.y-1,w.x.y),w.y), Interval2D(w.x,Interval(w.y.x,w.y.x+1)), Interval2D(w.x,Interval(w.y.y-1,w.y.y))]
		elseif (w.x.x < w.x.y-1) || (w.y.x < w.y.y-1)
			[w]
		else Interval2D[]
	end
	extr = map(w->yieldRepr(test_operator, _ReprMax(w), channel), reprs)
	maximum([extr..., typemin(T)])
end
WExtremeModal(test_operator::_TestOpLes, w::Interval2D, r::_Topo_TPP, channel::MatricialChannel{T,2}) where {T} = begin
	reprs = if (w.x.x < w.x.y-1) && (w.y.x < w.y.y-1)
			[Interval2D(Interval(w.x.x,w.x.x+1),w.y), Interval2D(Interval(w.x.y-1,w.x.y),w.y), Interval2D(w.x,Interval(w.y.x,w.y.x+1)), Interval2D(w.x,Interval(w.y.y-1,w.y.y))]
		elseif (w.x.x < w.x.y-1) || (w.y.x < w.y.y-1)
			[w]
		else Interval2D[]
	end
	extr = map(w->yieldRepr(test_operator, _ReprMin(w), channel), reprs)
	minimum([extr..., typemax(T)])
end

# TPPi: check 4 possible extensions of the box and perform a minimize+maximize step
WExtremaModal(test_operator::_TestOpGeq, w::Interval2D, r::_Topo_TPPi, channel::MatricialChannel{T,2}) where {T} = begin
	X,Y = size(channel)
	reprs = [
		((w.x.x-1 >= 1)   ? [Interval2D(Interval(w.x.x-1,w.x.y),w.y)] : Interval2D[])...,
		((w.x.y+1 <= X+1) ? [Interval2D(Interval(w.x.x,w.x.y+1),w.y)] : Interval2D[])...,
		((w.y.x-1 >= 1)   ? [Interval2D(w.x,Interval(w.y.x-1,w.y.y))] : Interval2D[])...,
		((w.y.y+1 <= Y+1) ? [Interval2D(w.x,Interval(w.y.x,w.y.y+1))] : Interval2D[])...,
	]
	extr = map(w->yieldReprs(test_operator, _ReprMin(w), channel), reprs)
	maxExtrema(extr)
end
WExtremeModal(test_operator::_TestOpGeq, w::Interval2D, r::_Topo_TPPi, channel::MatricialChannel{T,2}) where {T} = begin
	X,Y = size(channel)
	reprs = [
		((w.x.x-1 >= 1)   ? [Interval2D(Interval(w.x.x-1,w.x.y),w.y)] : Interval2D[])...,
		((w.x.y+1 <= X+1) ? [Interval2D(Interval(w.x.x,w.x.y+1),w.y)] : Interval2D[])...,
		((w.y.x-1 >= 1)   ? [Interval2D(w.x,Interval(w.y.x-1,w.y.y))] : Interval2D[])...,
		((w.y.y+1 <= Y+1) ? [Interval2D(w.x,Interval(w.y.x,w.y.y+1))] : Interval2D[])...,
	]
	extr = map(w->yieldRepr(test_operator, _ReprMin(w), channel), reprs)
	maximum([extr..., typemin(T)])
end
WExtremeModal(test_operator::_TestOpLes, w::Interval2D, r::_Topo_TPPi, channel::MatricialChannel{T,2}) where {T} = begin
	X,Y = size(channel)
	reprs = [
		((w.x.x-1 >= 1)   ? [Interval2D(Interval(w.x.x-1,w.x.y),w.y)] : Interval2D[])...,
		((w.x.y+1 <= X+1) ? [Interval2D(Interval(w.x.x,w.x.y+1),w.y)] : Interval2D[])...,
		((w.y.x-1 >= 1)   ? [Interval2D(w.x,Interval(w.y.x-1,w.y.y))] : Interval2D[])...,
		((w.y.y+1 <= Y+1) ? [Interval2D(w.x,Interval(w.y.x,w.y.y+1))] : Interval2D[])...,
	]
	extr = map(w->yieldRepr(test_operator, _ReprMax(w), channel), reprs)
	minimum([extr..., typemax(T)])
end

enumAccRepr(test_operator::TestOperator, w::Interval2D, ::_Topo_NTPP,  X::Integer, Y::Integer) = enumAccRepr(test_operator, w, _IA2DRel(IA_D,IA_D), X, Y)
enumAccRepr(test_operator::TestOperator, w::Interval2D, ::_Topo_NTPPi, X::Integer, Y::Integer) = enumAccRepr(test_operator, w, _IA2DRel(IA_Di,IA_Di), X, Y)

#=


# To test optimizations
fn1 = ModalLogic.enumAccRepr
fn2 = ModalLogic.enumAccRepr2
rel = ModalLogic.Topo_EC
X = 4
Y = 3
while(true)
	a = randn(4,4);
	wextr = (x)->ModalLogic.WExtrema([ModalLogic.TestOpGeq, ModalLogic.TestOpLes], x,a);
	# TODO try all rectangles, avoid randominzing like this... Also try all channel sizes
	x1 = rand(1:X);
	x2 = x1+rand(1:(X+1-x1));
	x3 = rand(1:Y);
	x4 = x3+rand(1:(Y+1-x3));
	for i in 1:X
		println(a[i,:]);
	end
	println(x1,",",x2);
	println(x3,",",x4);
	println(a[x1:x2-1,x3:x4-1]);
	print("[")
	print(fn1(ModalLogic.Interval2D((x1,x2),(x3,x4)), rel, size(a)...) |> (y)->map((x)->ModalLogic.print_world(x),y));
	println("]")
	print("[")
	print(fn2(ModalLogic.Interval2D((x1,x2),(x3,x4)), rel, size(a)...) |> (y)->map((x)->ModalLogic.print_world(x),y));
	println("]")
	println(fn1(ModalLogic.Interval2D((x1,x2),(x3,x4)), rel, size(a)...) |> wextr);
	println(fn2(ModalLogic.Interval2D((x1,x2),(x3,x4)), rel, size(a)...) |> wextr);
	(fn1(ModalLogic.Interval2D((x1,x2),(x3,x4)), rel, size(a)...) |> wextr) == (fn2(ModalLogic.Interval2D((x1,x2),(x3,x4)), rel, size(a)...) |> wextr) || break;
end

fn1 = ModalLogic.enumAccRepr
fn2 = ModalLogic.enumAccRepr2
rel = ModalLogic.Topo_EC
a = [253 670 577; 569 730 931; 633 850 679];
X,Y = size(a)
while(true)
	wextr = (x)->ModalLogic.WExtrema([ModalLogic.TestOpGeq, ModalLogic.TestOpLes], x,a);
	# TODO try all rectangles, avoid randominzing like this... Also try all channel sizes
	x1 = rand(1:X);
	x2 = x1+rand(1:(X+1-x1));
	x3 = rand(1:Y);
	x4 = x3+rand(1:(Y+1-x3));
	for i in 1:X
		println(a[i,:]);
	end
	println(x1,",",x2);
	println(x3,",",x4);
	println(a[x1:x2-1,x3:x4-1]);
	print("[")
	print(fn1(ModalLogic.Interval2D((x1,x2),(x3,x4)), rel, size(a)...) |> (y)->map((x)->ModalLogic.print_world(x),y));
	println("]")
	print("[")
	print(fn2(ModalLogic.Interval2D((x1,x2),(x3,x4)), rel, size(a)...) |> (y)->map((x)->ModalLogic.print_world(x),y));
	println("]")
	println(fn1(ModalLogic.Interval2D((x1,x2),(x3,x4)), rel, size(a)...) |> wextr);
	println(fn2(ModalLogic.Interval2D((x1,x2),(x3,x4)), rel, size(a)...) |> wextr);
	(fn1(ModalLogic.Interval2D((x1,x2),(x3,x4)), rel, size(a)...) |> wextr) == (fn2(ModalLogic.Interval2D((x1,x2),(x3,x4)), rel, size(a)...) |> wextr) || break;
end

fn1 = ModalLogic.enumAccRepr
fn2 = ModalLogic.enumAccRepr2
rel = ModalLogic.Topo_EC
a = [253 670 577; 569 730 931; 633 850 679];
X,Y = size(a)
while(true)
wextr = (x)->ModalLogic.WExtrema([ModalLogic.TestOpGeq, ModalLogic.TestOpLes], x,a);
# TODO try all rectangles, avoid randominzing like this... Also try all channel sizes
x1 = 2
x2 = 3
x3 = 2
x4 = 3
for i in 1:X
	println(a[i,:]);
end
println(x1,",",x2);
println(x3,",",x4);
println(a[x1:x2-1,x3:x4-1]);
print("[")
print(fn1(ModalLogic.Interval2D((x1,x2),(x3,x4)), rel, size(a)...) |> (y)->map((x)->ModalLogic.print_world(x),y));
println("]")
print("[")
print(fn2(ModalLogic.Interval2D((x1,x2),(x3,x4)), rel, size(a)...) |> (y)->map((x)->ModalLogic.print_world(x),y));
println("]")
println(fn1(ModalLogic.Interval2D((x1,x2),(x3,x4)), rel, size(a)...) |> wextr);
println(fn2(ModalLogic.Interval2D((x1,x2),(x3,x4)), rel, size(a)...) |> wextr);
(fn1(ModalLogic.Interval2D((x1,x2),(x3,x4)), rel, size(a)...) |> wextr) == (fn2(ModalLogic.Interval2D((x1,x2),(x3,x4)), rel, size(a)...) |> wextr) || break;
end

=#

################################################################################
# END Topological relations
################################################################################

# TODO Point
# ################################################################################
# # BEGIN Point
# ################################################################################

# struct Point <: AbstractWorld
	# Point(w::Point) = new(w.x,w.y)
# 	x :: Integer
# 	# TODO check x<=N but only in debug mode
# 	# Point(x) = x<=N ... ? new(x) : error("Can't instantiate Point(x={$x})")
# 	Point(x::Integer) = new(x)
# 	Point(::_emptyWorld) = new(0)
# 	Point(::_firstWorld) = new(1)
# 	Point(::_centeredWorld, X::Integer) = new(div(X,2)+1)
# end

# show(io::IO, r::Interval) = print(io, "($(x)×$(y))")

# enumAccBare(w::Point, ::_RelationId, XYZ::Vararg{Integer,N}) where N = [(w.x,)]
# enumAcc(S::AbstractWorldSet{Point}, r::_RelationAll, X::Integer) =
# 	IterTools.imap(Point, 1:X)

# worldTypeSize(::Type{Interval}) = 1

# @inline readWorld(w::Point, channel::MatricialChannel{T,1}) where {T} = channel[w.x]

# ################################################################################
# # END Point
# ################################################################################


export genericIntervalOntology,
				IntervalOntology,
				Interval2DOntology,
				getIntervalOntologyOfDim,
				genericIntervalTopologicalOntology,
				IntervalTopologicalOntology,
				Interval2DTopologicalOntology,
				getIntervalTopologicalOntologyOfDim

abstract type OntologyType end
struct _genericIntervalOntology  <: OntologyType end; const genericIntervalOntology  = _genericIntervalOntology();  # After
const IntervalOntology   = Ontology(Interval,IARelations)
const Interval2DOntology = Ontology(Interval2D,IA2DRelations)

struct _genericIntervalTopologicalOntology  <: OntologyType end; const genericIntervalTopologicalOntology  = _genericIntervalTopologicalOntology();  # After
const IntervalTopologicalOntology   = Ontology(Interval,TopoRelations)
const Interval2DTopologicalOntology = Ontology(Interval2D,TopoRelations)

getIntervalOntologyOfDim(::MatricialDataset{T,D}) where {T,D} = getIntervalOntologyOfDim(Val(D-2))
getIntervalOntologyOfDim(::Val{1}) = IntervalOntology
getIntervalOntologyOfDim(::Val{2}) = Interval2DOntology

getIntervalTopologicalOntologyOfDim(::MatricialDataset{T,D}) where {T,D} = getIntervalTopologicalOntologyOfDim(Val(D-2))
getIntervalTopologicalOntologyOfDim(::Val{1}) = IntervalTopologicalOntology
getIntervalTopologicalOntologyOfDim(::Val{2}) = Interval2DTopologicalOntology
