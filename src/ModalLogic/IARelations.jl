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

goesWith(::Type{Interval}, ::R where R<:_IARel) = true

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
enumAccRepr(test_operator::Union{_TestOpGeq,_TestOpLeq}, w::Interval, ::_IA_A,  X::Integer) = (w.y < X+1)                 ? _ReprVal(Interval(w.y, w.y+1)   ) : _ReprNone{Interval}() # [Interval(w.y, X+1)]     : Interval[]
enumAccRepr(test_operator::Union{_TestOpGeq,_TestOpLeq}, w::Interval, ::_IA_Ai, X::Integer) = (1 < w.x)                   ? _ReprVal(Interval(w.x-1, w.x)   ) : _ReprNone{Interval}() # [Interval(1, w.x)]       : Interval[]
enumAccRepr(test_operator::Union{_TestOpGeq,_TestOpLeq}, w::Interval, ::_IA_B,  X::Integer) = (w.x < w.y-1)               ? _ReprVal(Interval(w.x, w.x+1)   ) : _ReprNone{Interval}() # [Interval(w.x, w.y-1)]   : Interval[]
enumAccRepr(test_operator::Union{_TestOpGeq,_TestOpLeq}, w::Interval, ::_IA_E,  X::Integer) = (w.x+1 < w.y)               ? _ReprVal(Interval(w.y-1, w.y)   ) : _ReprNone{Interval}() # [Interval(w.x+1, w.y)]   : Interval[]

enumAccRepr(test_operator::_TestOpGeq, w::Interval, ::_IA_L,  X::Integer) = (w.y+1 < X+1)               ? _ReprMax(Interval(w.y+1, X+1)   ) : _ReprNone{Interval}() # [Interval(w.y+1, X+1)]   : Interval[]
enumAccRepr(test_operator::_TestOpGeq, w::Interval, ::_IA_Li, X::Integer) = (1 < w.x-1)                 ? _ReprMax(Interval(1, w.x-1)     ) : _ReprNone{Interval}() # [Interval(1, w.x-1)]     : Interval[]
enumAccRepr(test_operator::_TestOpGeq, w::Interval, ::_IA_D,  X::Integer) = (w.x+1 < w.y-1)             ? _ReprMax(Interval(w.x+1, w.y-1) ) : _ReprNone{Interval}() # [Interval(w.x+1, w.y-1)] : Interval[]
enumAccRepr(test_operator::_TestOpLeq, w::Interval, ::_IA_L,  X::Integer) = (w.y+1 < X+1)               ? _ReprMin(Interval(w.y+1, X+1)   ) : _ReprNone{Interval}() # [Interval(w.y+1, X+1)]   : Interval[]
enumAccRepr(test_operator::_TestOpLeq, w::Interval, ::_IA_Li, X::Integer) = (1 < w.x-1)                 ? _ReprMin(Interval(1, w.x-1)     ) : _ReprNone{Interval}() # [Interval(1, w.x-1)]     : Interval[]
enumAccRepr(test_operator::_TestOpLeq, w::Interval, ::_IA_D,  X::Integer) = (w.x+1 < w.y-1)             ? _ReprMin(Interval(w.x+1, w.y-1) ) : _ReprNone{Interval}() # [Interval(w.x+1, w.y-1)] : Interval[]

enumAccRepr(test_operator::_TestOpGeq, w::Interval, ::_IA_Bi, X::Integer) = (w.y < X+1)                 ? _ReprMin(Interval(w.x, w.y+1)   ) : _ReprNone{Interval}() # [Interval(w.x, X+1)]     : Interval[]
enumAccRepr(test_operator::_TestOpGeq, w::Interval, ::_IA_Ei, X::Integer) = (1 < w.x)                   ? _ReprMin(Interval(w.x-1, w.y)   ) : _ReprNone{Interval}() # [Interval(1, w.y)]       : Interval[]
enumAccRepr(test_operator::_TestOpGeq, w::Interval, ::_IA_Di, X::Integer) = (1 < w.x && w.y < X+1)      ? _ReprMin(Interval(w.x-1, w.y+1) ) : _ReprNone{Interval}() # [Interval(1, X+1)]       : Interval[]
enumAccRepr(test_operator::_TestOpGeq, w::Interval, ::_IA_O,  X::Integer) = (w.x+1 < w.y && w.y < X+1)  ? _ReprMin(Interval(w.y-1, w.y+1) ) : _ReprNone{Interval}() # [Interval(w.x+1, X+1)]   : Interval[]
enumAccRepr(test_operator::_TestOpGeq, w::Interval, ::_IA_Oi, X::Integer) = (1 < w.x && w.x+1 < w.y)    ? _ReprMin(Interval(w.x-1, w.x+1) ) : _ReprNone{Interval}() # [Interval(1, w.y-1)]     : Interval[]
enumAccRepr(test_operator::_TestOpLeq, w::Interval, ::_IA_Bi, X::Integer) = (w.y < X+1)                 ? _ReprMax(Interval(w.x, w.y+1)   ) : _ReprNone{Interval}() # [Interval(w.x, X+1)]     : Interval[]
enumAccRepr(test_operator::_TestOpLeq, w::Interval, ::_IA_Ei, X::Integer) = (1 < w.x)                   ? _ReprMax(Interval(w.x-1, w.y)   ) : _ReprNone{Interval}() # [Interval(1, w.y)]       : Interval[]
enumAccRepr(test_operator::_TestOpLeq, w::Interval, ::_IA_Di, X::Integer) = (1 < w.x && w.y < X+1)      ? _ReprMax(Interval(w.x-1, w.y+1) ) : _ReprNone{Interval}() # [Interval(1, X+1)]       : Interval[]
enumAccRepr(test_operator::_TestOpLeq, w::Interval, ::_IA_O,  X::Integer) = (w.x+1 < w.y && w.y < X+1)  ? _ReprMax(Interval(w.y-1, w.y+1) ) : _ReprNone{Interval}() # [Interval(w.x+1, X+1)]   : Interval[]
enumAccRepr(test_operator::_TestOpLeq, w::Interval, ::_IA_Oi, X::Integer) = (1 < w.x && w.x+1 < w.y)    ? _ReprMax(Interval(w.x-1, w.x+1) ) : _ReprNone{Interval}() # [Interval(1, w.y-1)]     : Interval[]

# WExtremaModal(test_operator::_TestOpGeq, w::Interval, ::_IA_A, channel::MatricialChannel{T,1}) where {T} =
# 	(w.y < length(channel)+1) ? (channel[w.y],channel[w.y]) : (typemax(T),typemin(T))
# WExtremeModal(test_operator::_TestOpGeq, w::Interval, ::_IA_A, channel::MatricialChannel{T,1}) where {T} =
# 	(w.y < length(channel)+1) ? channel[w.y] : typemax(T)
# WExtremeModal(test_operator::_TestOpLeq, w::Interval, ::_IA_A, channel::MatricialChannel{T,1}) where {T} =
# 	(w.y < length(channel)+1) ? channel[w.y] : typemin(T)

# WExtremaModal(test_operator::_TestOpGeq, w::Interval, ::_IA_Ai, channel::MatricialChannel{T,1}) where {T} =
# 	(1 < w.x) ? (channel[w.x-1],channel[w.x-1]) : (typemax(T),typemin(T))
# WExtremeModal(test_operator::_TestOpGeq, w::Interval, ::_IA_Ai, channel::MatricialChannel{T,1}) where {T} =
# 	(1 < w.x) ? channel[w.x-1] : typemax(T)
# WExtremeModal(test_operator::_TestOpLeq, w::Interval, ::_IA_Ai, channel::MatricialChannel{T,1}) where {T} =
# 	(1 < w.x) ? channel[w.x-1] : typemin(T)

# WExtremaModal(test_operator::_TestOpGeq, w::Interval, ::_IA_L, channel::MatricialChannel{T,1}) where {T} =
# 	(w.y+1 < length(channel)+1) ? reverse(extrema(channel[w.y+1:length(channel)])) : (typemax(T),typemin(T))
# WExtremeModal(test_operator::_TestOpGeq, w::Interval, ::_IA_L, channel::MatricialChannel{T,1}) where {T} =
# 	(w.y+1 < length(channel)+1) ? maximum(channel[w.y+1:length(channel)]) : typemax(T)
# WExtremeModal(test_operator::_TestOpLeq, w::Interval, ::_IA_L, channel::MatricialChannel{T,1}) where {T} =
# 	(w.y+1 < length(channel)+1) ? minumum(channel[w.y+1:length(channel)]) : typemin(T)

# WExtremaModal(test_operator::_TestOpGeq, w::Interval, ::_IA_Li, channel::MatricialChannel{T,1}) where {T} =
# 	(1 < w.x-1) ? reverse(extrema(channel[1:w.x-2])) : (typemax(T),typemin(T))
# WExtremeModal(test_operator::_TestOpGeq, w::Interval, ::_IA_Li, channel::MatricialChannel{T,1}) where {T} =
# 	(1 < w.x-1) ? maximum(channel[1:w.x-2]) : typemax(T)
# WExtremeModal(test_operator::_TestOpLeq, w::Interval, ::_IA_Li, channel::MatricialChannel{T,1}) where {T} =
# 	(1 < w.x-1) ? minumum(channel[1:w.x-2]) : typemin(T)

# WExtremaModal(test_operator::_TestOpGeq, w::Interval, ::_IA_B, channel::MatricialChannel{T,1}) where {T} =
# 	(w.x < w.y-1) ? (channel[w.x],channel[w.x]) : (typemax(T),typemin(T))
# WExtremeModal(test_operator::_TestOpGeq, w::Interval, ::_IA_B, channel::MatricialChannel{T,1}) where {T} =
# 	(w.x < w.y-1) ? channel[w.x] : typemax(T)
# WExtremeModal(test_operator::_TestOpLeq, w::Interval, ::_IA_B, channel::MatricialChannel{T,1}) where {T} =
# 	(w.x < w.y-1) ? channel[w.x] : typemin(T)

# WExtremaModal(test_operator::_TestOpGeq, w::Interval, ::_IA_Bi, channel::MatricialChannel{T,1}) where {T} =
# 	(w.y < length(channel)+1) ? (minimum(channel[w.x:w.y-1+1]),maximum(channel[w.x:w.y-1+1])) : (typemax(T),typemin(T))
# WExtremeModal(test_operator::_TestOpGeq, w::Interval, ::_IA_Bi, channel::MatricialChannel{T,1}) where {T} =
# 	(w.y < length(channel)+1) ? minimum(channel[w.x:w.y-1+1]) : typemax(T)
# WExtremeModal(test_operator::_TestOpLeq, w::Interval, ::_IA_Bi, channel::MatricialChannel{T,1}) where {T} =
# 	(w.y < length(channel)+1) ? maximum(channel[w.x:w.y-1+1]) : typemin(T)

# WExtremaModal(test_operator::_TestOpGeq, w::Interval, ::_IA_E, channel::MatricialChannel{T,1}) where {T} =
# 	(w.x+1 < w.y) ? (channel[w.y-1],channel[w.y-1]) : (typemax(T),typemin(T))
# WExtremeModal(test_operator::_TestOpGeq, w::Interval, ::_IA_E, channel::MatricialChannel{T,1}) where {T} =
# 	(w.x+1 < w.y) ? channel[w.y-1] : typemax(T)
# WExtremeModal(test_operator::_TestOpLeq, w::Interval, ::_IA_E, channel::MatricialChannel{T,1}) where {T} =
# 	(w.x+1 < w.y) ? channel[w.y-1] : typemin(T)

# WExtremaModal(test_operator::_TestOpGeq, w::Interval, ::_IA_Ei, channel::MatricialChannel{T,1}) where {T} =
# 	(1 < w.x) ? (minimum(channel[w.x-1:w.y-1]),maximum(channel[w.x-1:w.y-1])) : (typemax(T),typemin(T))
# WExtremeModal(test_operator::_TestOpGeq, w::Interval, ::_IA_Ei, channel::MatricialChannel{T,1}) where {T} =
# 	(1 < w.x) ? minimum(channel[w.x-1:w.y-1]) : typemax(T)
# WExtremeModal(test_operator::_TestOpLeq, w::Interval, ::_IA_Ei, channel::MatricialChannel{T,1}) where {T} =
# 	(1 < w.x) ? maximum(channel[w.x-1:w.y-1]) : typemin(T)

# WExtremaModal(test_operator::_TestOpGeq, w::Interval, ::_IA_D, channel::MatricialChannel{T,1}) where {T} =
# 	(w.x+1 < w.y-1) ? reverse(extrema(channel[w.x+1:w.y-1-1])) : (typemax(T),typemin(T))
# WExtremeModal(test_operator::_TestOpGeq, w::Interval, ::_IA_D, channel::MatricialChannel{T,1}) where {T} =
# 	(w.x+1 < w.y-1) ? maximum(channel[w.x+1:w.y-1-1]) : typemax(T)
# WExtremeModal(test_operator::_TestOpLeq, w::Interval, ::_IA_D, channel::MatricialChannel{T,1}) where {T} =
# 	(w.x+1 < w.y-1) ? minumum(channel[w.x+1:w.y-1-1]) : typemin(T)

# WExtremaModal(test_operator::_TestOpGeq, w::Interval, ::_IA_Di, channel::MatricialChannel{T,1}) where {T} =
# 	(1 < w.x && w.y < length(channel)+1) ? (minimum(channel[w.x-1:w.y-1+1]),maximum(channel[w.x-1:w.y-1+1])) : (typemax(T),typemin(T))
# WExtremeModal(test_operator::_TestOpGeq, w::Interval, ::_IA_Di, channel::MatricialChannel{T,1}) where {T} =
# 	(1 < w.x && w.y < length(channel)+1) ? minimum(channel[w.x-1:w.y-1+1]) : typemax(T)
# WExtremeModal(test_operator::_TestOpLeq, w::Interval, ::_IA_Di, channel::MatricialChannel{T,1}) where {T} =
# 	(1 < w.x && w.y < length(channel)+1) ? maximum(channel[w.x-1:w.y-1+1]) : typemin(T)

# WExtremaModal(test_operator::_TestOpGeq, w::Interval, ::_IA_O, channel::MatricialChannel{T,1}) where {T} =
# 	(w.x+1 < w.y && w.y < length(channel)+1) ? (minimum(channel[w.y-1:w.y-1+1]),maximum(channel[w.y-1:w.y-1+1])) : (typemax(T),typemin(T))
# WExtremeModal(test_operator::_TestOpGeq, w::Interval, ::_IA_O, channel::MatricialChannel{T,1}) where {T} =
# 	(w.x+1 < w.y && w.y < length(channel)+1) ? minimum(channel[w.y-1:w.y-1+1]) : typemax(T)
# WExtremeModal(test_operator::_TestOpLeq, w::Interval, ::_IA_O, channel::MatricialChannel{T,1}) where {T} =
# 	(w.x+1 < w.y && w.y < length(channel)+1) ? maximum(channel[w.y-1:w.y-1+1]) : typemin(T)

# WExtremaModal(test_operator::_TestOpGeq, w::Interval, ::_IA_Oi, channel::MatricialChannel{T,1}) where {T} =
# 	(1 < w.x && w.x+1 < w.y) ? (minimum(channel[w.x-1:w.x]),maximum(channel[w.x-1:w.x])) : (typemax(T),typemin(T))
# WExtremeModal(test_operator::_TestOpGeq, w::Interval, ::_IA_Oi, channel::MatricialChannel{T,1}) where {T} =
# 	(1 < w.x && w.x+1 < w.y) ? minimum(channel[w.x-1:w.x]) : typemax(T)
# WExtremeModal(test_operator::_TestOpLeq, w::Interval, ::_IA_Oi, channel::MatricialChannel{T,1}) where {T} =
# 	(1 < w.x && w.x+1 < w.y) ? maximum(channel[w.x-1:w.x]) : typemin(T)

################################################################################
# END IA relations
################################################################################
