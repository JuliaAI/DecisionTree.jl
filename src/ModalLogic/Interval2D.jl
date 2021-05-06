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

worldTypeDimensionality(::Type{Interval2D}) = 2

yieldReprs(test_operator::_TestOpGeq, repr::_ReprMax{Interval2D},  channel::MatricialChannel{T,2}) where {T} =
	reverse(extrema(readWorld(repr.w, channel)))::NTuple{2,T}
yieldReprs(test_operator::_TestOpGeq, repr::_ReprMin{Interval2D},  channel::MatricialChannel{T,2}) where {T} =
	extrema(readWorld(repr.w, channel))::NTuple{2,T}
yieldReprs(test_operator::_TestOpGeq, repr::_ReprVal{Interval2D},  channel::MatricialChannel{T,2}) where {T} =
	(channel[repr.w.x.x, repr.w.y.x],channel[repr.w.x.x, repr.w.y.x])::NTuple{2,T}
yieldReprs(test_operator::_TestOpGeq, repr::_ReprNone{Interval2D}, channel::MatricialChannel{T,2}) where {T} =
	(typemin(T),typemax(T))::NTuple{2,T}

yieldRepr(test_operator::Union{_TestOpGeq,_TestOpLeq}, repr::_ReprMax{Interval2D},  channel::MatricialChannel{T,2}) where {T} =
	maximum(readWorld(repr.w, channel))::T
yieldRepr(test_operator::Union{_TestOpGeq,_TestOpLeq}, repr::_ReprMin{Interval2D},  channel::MatricialChannel{T,2}) where {T} =
	minimum(readWorld(repr.w, channel))::T
yieldRepr(test_operator::Union{_TestOpGeq,_TestOpLeq}, repr::_ReprVal{Interval2D},  channel::MatricialChannel{T,2}) where {T} =
	channel[repr.w.x.x, repr.w.y.x]::T
yieldRepr(test_operator::_TestOpGeq, repr::_ReprNone{Interval2D}, channel::MatricialChannel{T,2}) where {T} =
	typemin(T)::T
yieldRepr(test_operator::_TestOpLeq, repr::_ReprNone{Interval2D}, channel::MatricialChannel{T,2}) where {T} =
	typemax(T)::T

enumAccRepr(test_operator::_TestOpGeq, w::Interval2D, ::_RelationId,  X::Integer, Y::Integer) = _ReprMin(w)
enumAccRepr(test_operator::_TestOpLeq, w::Interval2D, ::_RelationId,  X::Integer, Y::Integer) = _ReprMax(w)
enumAccRepr(test_operator::_TestOpGeq, w::Interval2D, ::_RelationAll, X::Integer, Y::Integer) = _ReprMax(Interval2D(Interval(1,X+1), Interval(1,Y+1)))
enumAccRepr(test_operator::_TestOpLeq, w::Interval2D, ::_RelationAll, X::Integer, Y::Integer) = _ReprMin(Interval2D(Interval(1,X+1), Interval(1,Y+1)))

# TODO write only one ExtremeModal/ExtremaModal
# TODO optimize relationAll
computeModalThresholdDual(test_operator::_TestOpGeq, w::Interval2D, r::R where R<:AbstractRelation, channel::MatricialChannel{T,2}) where {T} = begin
	# if (channel == [412 489 559 619 784; 795 771 1317 854 1256; 971 874 878 1278 560] && w.x.x==1 && w.x.y==3 && w.y.x==3 && w.y.y==4)
	# 	println(enumAccRepr(test_operator, w, r, size(channel)...))
	# 	readline()
	# end
	yieldReprs(test_operator, enumAccRepr(test_operator, w, r, size(channel)...), channel)
end
computeModalThreshold(test_operator::Union{_TestOpGeq,_TestOpLeq}, w::Interval2D, r::R where R<:AbstractRelation, channel::MatricialChannel{T,2}) where {T} =
	yieldRepr(test_operator, enumAccRepr(test_operator, w, r, size(channel)...), channel)
# channel = [1,2,3,2,8,349,0,830,7290,298,20,29,2790,27,90279,270,2722,79072,0]
# w = ModalLogic.Interval(3,9)
# # w = ModalLogic.Interval(3,4)
# for relation in ModalLogic.IARelations
# 	ModalLogic.computeModalThresholdDual(ModalLogic.TestOpGeq, w, relation, channel)
# end

# channel2 = randn(3,4)
# channel2[1:3,1]
# channel2[1:3,2]
# channel2[1:3,3]
# channel2[1:3,4]
# vals=channel2
# mapslices(maximum, vals, dims=1)

# computeModalThresholdDual(test_operator::_TestOpGeq, w::Interval2D, ::_RelationAll, channel::MatricialChannel{T,2}) where {T} = begin
# 	# X = size(channel, 1)
# 	# Y = size(channel, 2)
# 	# println("Check!")
# 	# println(test_operator)
# 	# println(w)
# 	# println(relation)
# 	# println(channel)
# 	# println(computePropositionalThresholdDual(test_operator, Interval2D(Interval(1,X+1), Interval(1, Y+1)), channel))
# 	# readline()
# 	# computePropositionalThresholdDual(test_operator, Interval2D(Interval(1,X+1), Interval(1, Y+1)), channel)
# 	reverse(extrema(channel))
# end
# computeModalThreshold(test_operator::_TestOpGeq, w::Interval2D, ::_RelationAll, channel::MatricialChannel{T,2}) where {T} = begin
# 	# TODO optimize this by replacing readworld with channel[1:X]...
# 	# X = size(channel, 1)
# 	# Y = size(channel, 2)
# 	# maximum(channel[1:X,1:Y])
# 	maximum(channel)
# end
# computeModalThreshold(test_operator::_TestOpLeq, w::Interval2D, ::_RelationAll, channel::MatricialChannel{T,2}) where {T} = begin
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
enumAccessibles(S::AbstractWorldSet{Interval2D}, r::_RelationAll, X::Integer, Y::Integer) =
	IterTools.imap(Interval2D,
		Iterators.product(enumPairsIn(1, X+1), enumPairsIn(1, Y+1))
		# enumAccBare(w..., IA2DRel(RelationAll,RelationAll), X, Y)
	)
	# IterTools.imap(Interval2D, enumPairsIn(1, X+1), enumPairsIn(1, Y+1))
		# enumAccBare(w, IA2DRel(RelationAll,RelationAll), X, Y)
# enumAccBare(w::Interval2D, r::_RelationAll, X::Integer, Y::Integer) =
# 	enumAccBare(w, _IA2DRel(RelationAll,RelationAll), X, Y)

# worldTypeSize(::Type{Interval2D}) = 4
n_worlds(::Type{Interval2D}, channel_size::Tuple{Integer,Integer}) = n_worlds(Interval, channel_size[1]) * n_worlds(Interval, channel_size[2])

print_world(w::Interval2D) = println("Interval2D [$(w.x.x),$(w.x.y)) × [$(w.y.x),$(w.y.y)), length $(w.x.y-w.x.x)×$(w.y.y-w.y.x) = $((w.x.y-w.x.x)*(w.y.y-w.y.x))")

@inline readWorld(w::Interval2D, channel::MatricialChannel{T,2}) where {T} = channel[w.x.x:w.x.y-1,w.y.x:w.y.y-1]
