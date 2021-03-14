
################################################################################
# BEGIN Topological relations
################################################################################

# Topological relations
abstract type _TopoRel <: AbstractRelation end
# Relations for RCC8
struct _Topo_DC     <: _TopoRel end; const Topo_DC     = _Topo_DC();     # Disconnected
struct _Topo_EC     <: _TopoRel end; const Topo_EC     = _Topo_EC();     # Externally connected
struct _Topo_PO     <: _TopoRel end; const Topo_PO     = _Topo_PO();     # Partially overlapping
struct _Topo_TPP    <: _TopoRel end; const Topo_TPP    = _Topo_TPP();    # Tangential proper part
struct _Topo_TPPi   <: _TopoRel end; const Topo_TPPi   = _Topo_TPPi();   # Tangential proper part inverse
struct _Topo_NTPP   <: _TopoRel end; const Topo_NTPP   = _Topo_NTPP();   # Non-tangential proper part
struct _Topo_NTPPi  <: _TopoRel end; const Topo_NTPPi  = _Topo_NTPPi();  # Non-tangential proper part inverse
# Coarser relations for RCC5
struct _Topo_DR     <: _TopoRel end; const Topo_DR     = _Topo_DR();     # Disjointed
struct _Topo_PP     <: _TopoRel end; const Topo_PP     = _Topo_PP();     # Proper part
struct _Topo_PPi    <: _TopoRel end; const Topo_PPi    = _Topo_PPi();    # Proper part inverse

display_rel_short(::_Topo_DC)    = "DC"
display_rel_short(::_Topo_EC)    = "EC"
display_rel_short(::_Topo_PO)    = "PO"
display_rel_short(::_Topo_TPP)   = "TPP"
display_rel_short(::_Topo_TPPi)  = "T̅P̅P̅" # T̄P̄P̄
display_rel_short(::_Topo_NTPP)  = "NTPP"
display_rel_short(::_Topo_NTPPi) = "N̅T̅P̅P̅" # N̄T̄P̄P̄

display_rel_short(::_Topo_DR)   = "DR"
display_rel_short(::_Topo_PP)   = "PP"
display_rel_short(::_Topo_PPi)  = "TP̅P̅"

const RCC8Relations = [Topo_DC, Topo_EC, Topo_PO, Topo_TPP, Topo_TPPi, Topo_NTPP, Topo_NTPPi]
const RCC5Relations = [Topo_DR, Topo_PO, Topo_PP, Topo_PPi]

topo2IARelations(::_Topo_DC)     = [IA_L,  IA_Li]
topo2IARelations(::_Topo_EC)     = [IA_A,  IA_Ai]
topo2IARelations(::_Topo_PO)     = [IA_O,  IA_Oi]
topo2IARelations(::_Topo_TPP)    = [IA_B,  IA_E]
topo2IARelations(::_Topo_TPPi)   = [IA_Bi, IA_Ei]
topo2IARelations(::_Topo_NTPP)   = [IA_D]
topo2IARelations(::_Topo_NTPPi)  = [IA_Di]

RCC52RCC8Relations(::_Topo_DR)    = [Topo_DC, Topo_EC]
RCC52RCC8Relations(::_Topo_PP)    = [Topo_TPP,  Topo_NTPP]
RCC52RCC8Relations(::_Topo_PPi)   = [Topo_TPPi,  Topo_NTPPi]

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
# WExtremeModal(test_operator::_TestOpLeq, w::Interval, r::Union{_Topo_DC,_Topo_EC,_Topo_PO,_Topo_TPP,_Topo_TPPi}, channel::MatricialChannel{T,1}) where {T} = begin
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
WExtremeModal(test_operator::_TestOpLeq, w::Interval, r::_Topo_DC, channel::MatricialChannel{T,1}) where {T} = begin
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
WExtremeModal(test_operator::_TestOpLeq, w::Interval, r::_Topo_EC, channel::MatricialChannel{T,1}) where {T} = begin
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
WExtremeModal(test_operator::_TestOpLeq, w::Interval, r::_Topo_PO, channel::MatricialChannel{T,1}) where {T} = begin
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
WExtremeModal(test_operator::_TestOpLeq, w::Interval, r::_Topo_TPP, channel::MatricialChannel{T,1}) where {T} = begin
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
WExtremeModal(test_operator::_TestOpLeq, w::Interval, r::_Topo_TPPi, channel::MatricialChannel{T,1}) where {T} = begin
	repr1 = enumAccRepr(test_operator, w, IA_Bi, length(channel))
	repr2 = enumAccRepr(test_operator, w, IA_Ei, length(channel))
	min(yieldRepr(test_operator, repr1, channel), yieldRepr(test_operator, repr2, channel))
end

enumAccRepr(test_operator::TestOperator, w::Interval, ::_Topo_NTPP,  X::Integer) = enumAccRepr(test_operator, w, IA_D, X)
enumAccRepr(test_operator::TestOperator, w::Interval, ::_Topo_NTPPi, X::Integer) = enumAccRepr(test_operator, w, IA_Di, X)

################################################################################
# END Topological relations
################################################################################
