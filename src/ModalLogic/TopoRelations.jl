
################################################################################
# BEGIN Topological relations
################################################################################

# Topological relations
abstract type _TopoRel <: AbstractRelation end
# Relations for RCC8
abstract type _TopoRelRCC8 <: _TopoRel end
struct _Topo_DC     <: _TopoRelRCC8 end; const Topo_DC     = _Topo_DC();     # Disconnected
struct _Topo_EC     <: _TopoRelRCC8 end; const Topo_EC     = _Topo_EC();     # Externally connected
struct _Topo_PO     <: _TopoRelRCC8 end; const Topo_PO     = _Topo_PO();     # Partially overlapping
struct _Topo_TPP    <: _TopoRelRCC8 end; const Topo_TPP    = _Topo_TPP();    # Tangential proper part
struct _Topo_TPPi   <: _TopoRelRCC8 end; const Topo_TPPi   = _Topo_TPPi();   # Tangential proper part inverse
struct _Topo_NTPP   <: _TopoRelRCC8 end; const Topo_NTPP   = _Topo_NTPP();   # Non-tangential proper part
struct _Topo_NTPPi  <: _TopoRelRCC8 end; const Topo_NTPPi  = _Topo_NTPPi();  # Non-tangential proper part inverse
# Coarser relations for RCC5
abstract type _TopoRelRCC5 <: _TopoRel end
struct _Topo_DR     <: _TopoRelRCC5 end; const Topo_DR     = _Topo_DR();     # Disjointed
struct _Topo_PP     <: _TopoRelRCC5 end; const Topo_PP     = _Topo_PP();     # Proper part
struct _Topo_PPi    <: _TopoRelRCC5 end; const Topo_PPi    = _Topo_PPi();    # Proper part inverse

goesWith(::Type{Interval}, ::R where R<:_TopoRel) = true


display_rel_short(::_Topo_DC)    = "DC"
display_rel_short(::_Topo_EC)    = "EC"
display_rel_short(::_Topo_PO)    = "PO"
display_rel_short(::_Topo_TPP)   = "TPP"
display_rel_short(::_Topo_TPPi)  = "T̅P̅P̅" # T̄P̄P̄
display_rel_short(::_Topo_NTPP)  = "NTPP"
display_rel_short(::_Topo_NTPPi) = "N̅T̅P̅P̅" # N̄T̄P̄P̄

display_rel_short(::_Topo_DR)   = "DR"
display_rel_short(::_Topo_PP)   = "PP"
display_rel_short(::_Topo_PPi)  = "P̅P̅"

const RCC8Relations = [Topo_DC, Topo_EC, Topo_PO, Topo_TPP, Topo_TPPi, Topo_NTPP, Topo_NTPPi]
const RCC5Relations = [Topo_DR, Topo_PO, Topo_PP, Topo_PPi]

const _TopoRelRCC8FromIA = Union{_Topo_DC,_Topo_EC,_Topo_PO,_Topo_TPP,_Topo_TPPi}

topo2IARelations(::_Topo_DC)     = [IA_L,  IA_Li]
topo2IARelations(::_Topo_EC)     = [IA_A,  IA_Ai]
topo2IARelations(::_Topo_PO)     = [IA_O,  IA_Oi]
topo2IARelations(::_Topo_TPP)    = [IA_B,  IA_E]
topo2IARelations(::_Topo_TPPi)   = [IA_Bi, IA_Ei]
topo2IARelations(::_Topo_NTPP)   = [IA_D]
topo2IARelations(::_Topo_NTPPi)  = [IA_Di]

# TODO RCC5 can be better written as a combination of IA7 relations!
RCC52RCC8Relations(::_Topo_DR)    = [Topo_DC, Topo_EC]
RCC52RCC8Relations(::_Topo_PP)    = [Topo_TPP,  Topo_NTPP]
RCC52RCC8Relations(::_Topo_PPi)   = [Topo_TPPi,  Topo_NTPPi]

# Enumerate accessible worlds from a single world
enumAccBare(w::Interval, r::R where R<:_TopoRelRCC8FromIA,    X::Integer) = Iterators.flatten((enumAccBare(w, IA_r,  X) for IA_r in topo2IARelations(r)))
# enumAccBare(w::Interval, ::_Topo_DC,    X::Integer) = Iterators.flatten((enumAccBare(w, IA_L,  X), enumAccBare(w, IA_Li, X)))
# enumAccBare(w::Interval, ::_Topo_EC,    X::Integer) = Iterators.flatten((enumAccBare(w, IA_A,  X), enumAccBare(w, IA_Ai, X)))
# enumAccBare(w::Interval, ::_Topo_PO,    X::Integer) = Iterators.flatten((enumAccBare(w, IA_O,  X), enumAccBare(w, IA_Oi, X)))
# enumAccBare(w::Interval, ::_Topo_TPP,   X::Integer) = Iterators.flatten((enumAccBare(w, IA_B,  X), enumAccBare(w, IA_E,  X)))
# enumAccBare(w::Interval, ::_Topo_TPPi,  X::Integer) = Iterators.flatten((enumAccBare(w, IA_Bi, X), enumAccBare(w, IA_Ei, X)))
enumAccBare(w::Interval, ::_Topo_NTPP,  X::Integer) = enumAccBare(w, IA_D, X)
enumAccBare(w::Interval, ::_Topo_NTPPi, X::Integer) = enumAccBare(w, IA_Di, X)

# RCC5 computed as a combination. TODO could actually be written more lightly as the combination of IA7 relations?
enumAccBare(w::Interval, r::R where R<:_TopoRelRCC5,  XYZ::Vararg{Integer,1}) =
	Iterators.flatten((enumAccBare(w, IA_r,  XYZ...) for RCC8_r in RCC52RCC8Relations(r) for IA_r in topo2IARelations(RCC8_r)))

WExtremaModal(test_operator::_TestOpGeq, w::Interval, r::R where R<:_TopoRelRCC8FromIA, channel::MatricialChannel{T,1}) where {T} = begin
	maxExtrema(
		map((IA_r)->(yieldReprs(test_operator, enumAccRepr(test_operator, w, IA_r, length(channel)), channel)), topo2IARelations(r))
	)
end
WExtremeModal(test_operator::_TestOpGeq, w::Interval, r::R where R<:_TopoRelRCC8FromIA, channel::MatricialChannel{T,1}) where {T} = begin
	maximum(
		map((IA_r)->(yieldRepr(test_operator, enumAccRepr(test_operator, w, IA_r, length(channel)), channel)), topo2IARelations(r))
	)
end
WExtremeModal(test_operator::_TestOpLeq, w::Interval, r::R where R<:_TopoRelRCC8FromIA, channel::MatricialChannel{T,1}) where {T} = begin
	mininimum(
		map((IA_r)->(yieldRepr(test_operator, enumAccRepr(test_operator, w, IA_r, length(channel)), channel)), topo2IARelations(r))
	)
end

enumAccRepr(test_operator::TestOperator, w::Interval, ::_Topo_NTPP,  X::Integer) = enumAccRepr(test_operator, w, IA_D, X)
enumAccRepr(test_operator::TestOperator, w::Interval, ::_Topo_NTPPi, X::Integer) = enumAccRepr(test_operator, w, IA_Di, X)

WExtremaModal(test_operator::_TestOpGeq, w::Interval, r::R where R<:_TopoRelRCC5, channel::MatricialChannel{T,1}) where {T} = begin
	maxExtrema(
		map((IA_r)->(yieldReprs(test_operator, enumAccRepr(test_operator, w, IA_r, size(channel)...), channel)), [IA_r for RCC8_r in RCC52RCC8Relations(r) for IA_r in topo2IARelations(RCC8_r)])
	)
end
WExtremeModal(test_operator::_TestOpGeq, w::Interval, r::R where R<:_TopoRelRCC5, channel::MatricialChannel{T,1}) where {T} = begin
	maximum(
		map((IA_r)->(yieldRepr(test_operator, enumAccRepr(test_operator, w, IA_r, size(channel)...), channel)), [IA_r for RCC8_r in RCC52RCC8Relations(r) for IA_r in topo2IARelations(RCC8_r)])
	)
end
WExtremeModal(test_operator::_TestOpLeq, w::Interval, r::R where R<:_TopoRelRCC5, channel::MatricialChannel{T,1}) where {T} = begin
	mininimum(
		map((IA_r)->(yieldRepr(test_operator, enumAccRepr(test_operator, w, IA_r, size(channel)...), channel)), [IA_r for RCC8_r in RCC52RCC8Relations(r) for IA_r in topo2IARelations(RCC8_r)])
	)
end

################################################################################
# END Topological relations
################################################################################
