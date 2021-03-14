
################################################################################
# BEGIN 2D Topological relations
################################################################################

goesWith(::Type{Interval2D}, ::_TopoRel) = true

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
enumAccRepr(test_operator::_TestOpLeq, w::Interval, ::_Virtual_Enlarge,  X::Integer) = _ReprMax(enlargeInterval(w,X))


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
			 yieldRepr(test_operator, repry1, channel),
			 yieldRepr(test_operator, repry2, channel))
end
WExtremeModal(test_operator::_TestOpLeq, w::Interval2D, r::_Topo_DC, channel::MatricialChannel{T,2}) where {T} = begin
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
WExtremeModal(test_operator::_TestOpLeq, w::Interval2D, r::_Topo_EC, channel::MatricialChannel{T,2}) where {T} = begin
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

	maxExtrema(
		yieldMinMaxCombinations(test_operator, enumAccRepr2D(test_operator, w, RelationId, ry1,    size(channel)..., _ReprFake), channel, 2),
		yieldMinMaxCombinations(test_operator, enumAccRepr2D(test_operator, w, RelationId, ry2,    size(channel)..., _ReprFake), channel, 2),
		yieldMinMaxCombinations(test_operator, enumAccRepr2D(test_operator, w, rx1,    RelationId, size(channel)..., _ReprFake), channel, 1),
		yieldMinMaxCombinations(test_operator, enumAccRepr2D(test_operator, w, rx2,    RelationId, size(channel)..., _ReprFake), channel, 1),
	)
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

	max(
		yieldMinMaxCombination(test_operator, enumAccRepr2D(test_operator, w, RelationId, ry1,    size(channel)..., _ReprFake), channel, 2),
		yieldMinMaxCombination(test_operator, enumAccRepr2D(test_operator, w, RelationId, ry2,    size(channel)..., _ReprFake), channel, 2),
		yieldMinMaxCombination(test_operator, enumAccRepr2D(test_operator, w, rx1,    RelationId, size(channel)..., _ReprFake), channel, 1),
		yieldMinMaxCombination(test_operator, enumAccRepr2D(test_operator, w, rx2,    RelationId, size(channel)..., _ReprFake), channel, 1),
	)
end
WExtremeModal(test_operator::_TestOpLeq, w::Interval2D, r::_Topo_PO, channel::MatricialChannel{T,2}) where {T} = begin
	x_singleton = ! (w.x.x < w.x.y-1)
	y_singleton = ! (w.y.x < w.y.y-1)
	if x_singleton && y_singleton
		return typemax(T)
	end
	
	rx1,rx2 = x_singleton ? (IA_Bi,IA_Ei) : (IA_O,IA_Oi)
	ry1,ry2 = y_singleton ? (IA_Bi,IA_Ei) : (IA_O,IA_Oi)

	min(
		yieldMinMaxCombination(test_operator, enumAccRepr2D(test_operator, w, RelationId, ry1,    size(channel)..., _ReprFake), channel, 2),
		yieldMinMaxCombination(test_operator, enumAccRepr2D(test_operator, w, RelationId, ry2,    size(channel)..., _ReprFake), channel, 2),
		yieldMinMaxCombination(test_operator, enumAccRepr2D(test_operator, w, rx1,    RelationId, size(channel)..., _ReprFake), channel, 1),
		yieldMinMaxCombination(test_operator, enumAccRepr2D(test_operator, w, rx2,    RelationId, size(channel)..., _ReprFake), channel, 1),
	)
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
WExtremeModal(test_operator::_TestOpLeq, w::Interval2D, r::_Topo_TPP, channel::MatricialChannel{T,2}) where {T} = begin
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
WExtremeModal(test_operator::_TestOpLeq, w::Interval2D, r::_Topo_TPPi, channel::MatricialChannel{T,2}) where {T} = begin
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
	wextr = (x)->ModalLogic.WExtrema([ModalLogic.TestOpGeq, ModalLogic.TestOpLeq], x,a);
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
	wextr = (x)->ModalLogic.WExtrema([ModalLogic.TestOpGeq, ModalLogic.TestOpLeq], x,a);
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
wextr = (x)->ModalLogic.WExtrema([ModalLogic.TestOpGeq, ModalLogic.TestOpLeq], x,a);
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
# END 2D Topological relations
################################################################################
