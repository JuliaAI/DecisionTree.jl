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

# Note: with Points, < and >= is redundant

# show(io::IO, r::Interval) = print(io, "($(x)×$(y))")

# enumAccBare(w::Point, ::_RelationId, XYZ::Vararg{Integer,N}) where N = [(w.x,)]
# enumAccessibles(S::AbstractWorldSet{Point}, r::_RelationAll, X::Integer) =
# 	IterTools.imap(Point, 1:X)

# worldTypeSize(::Type{Interval}) = 1

# @inline readWorld(w::Point, channel::MatricialChannel{T,1}) where {T} = channel[w.x]

# ################################################################################
# # END Point
# ################################################################################
