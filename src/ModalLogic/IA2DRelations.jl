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

@inline enumAccRepr2D(test_operator::TestOperator, w::Interval2D, rx::R1 where R1<:AbstractRelation, ry::R2 where R2<:AbstractRelation, X::Integer, Y::Integer, _ReprConstructor::Type{rT}) where {rT<:_ReprTreatment} = begin
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

enumAccRepr(test_operator::_TestOpLeq, w::Interval2D, r::_IA2DRel{R1,R2} where {R1<:_IA2DRelMaximizer,R2<:_IA2DRelMaximizer}, X::Integer, Y::Integer) =
	enumAccRepr2D(test_operator, w, r.x, r.y, X, Y, _ReprMin)
enumAccRepr(test_operator::_TestOpLeq, w::Interval2D, r::_IA2DRel{R1,R2} where {R1<:_IA2DRelMinimizer,R2<:_IA2DRelMinimizer}, X::Integer, Y::Integer) =
	enumAccRepr2D(test_operator, w, r.x, r.y, X, Y, _ReprMax)
enumAccRepr(test_operator::_TestOpLeq, w::Interval2D, r::_IA2DRel{R1,R2} where {R1<:_IA2DRelSingleVal,R2<:_IA2DRelSingleVal}, X::Integer, Y::Integer) =
	enumAccRepr2D(test_operator, w, r.x, r.y, X, Y, _ReprVal)

enumAccRepr(test_operator::_TestOpLeq, w::Interval2D, r::_IA2DRel{R1,R2} where {R1<:_IA2DRelMaximizer,R2<:_IA2DRelSingleVal}, X::Integer, Y::Integer) =
	enumAccRepr2D(test_operator, w, r.x, r.y, X, Y, _ReprMin)
enumAccRepr(test_operator::_TestOpLeq, w::Interval2D, r::_IA2DRel{R1,R2} where {R1<:_IA2DRelMinimizer,R2<:_IA2DRelSingleVal}, X::Integer, Y::Integer) =
	enumAccRepr2D(test_operator, w, r.x, r.y, X, Y, _ReprMax)
enumAccRepr(test_operator::_TestOpLeq, w::Interval2D, r::_IA2DRel{R1,R2} where {R1<:_IA2DRelSingleVal,R2<:_IA2DRelMaximizer}, X::Integer, Y::Integer) =
	enumAccRepr2D(test_operator, w, r.x, r.y, X, Y, _ReprMin)
enumAccRepr(test_operator::_TestOpLeq, w::Interval2D, r::_IA2DRel{R1,R2} where {R1<:_IA2DRelSingleVal,R2<:_IA2DRelMinimizer}, X::Integer, Y::Integer) =
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

yieldMinMaxCombination(test_operator::_TestOpLeq, productRepr::_ReprTreatment, channel::MatricialChannel{T,2}, dims::Integer) where {T} = begin
	if productRepr == _ReprNone{Interval2D}()
		return typemax(T)
	end
	vals = readWorld(productRepr.w, channel)
	minimum(mapslices(maximum, vals, dims=dims))
end

WExtremaModal(test_operator::_TestOpGeq, w::Interval2D, r::_IA2DRel{R1,R2} where {R1<:_IA2DRelMinimizer,R2<:_IA2DRelMaximizer}, channel::MatricialChannel{T,2}) where {T} = begin
	yieldMinMaxCombinations(test_operator, enumAccRepr2D(test_operator, w, r.x, r.y, size(channel)..., _ReprFake), channel, 1)
end
WExtremeModal(test_operator::Union{_TestOpGeq,_TestOpLeq}, w::Interval2D, r::_IA2DRel{R1,R2} where {R1<:_IA2DRelMinimizer,R2<:_IA2DRelMaximizer}, channel::MatricialChannel{T,2}) where {T} = begin
	yieldMinMaxCombination(test_operator, enumAccRepr2D(test_operator, w, r.x, r.y, size(channel)..., _ReprFake), channel, 1)
end
WExtremaModal(test_operator::_TestOpGeq, w::Interval2D, r::_IA2DRel{R1,R2} where {R1<:_IA2DRelMaximizer,R2<:_IA2DRelMinimizer}, channel::MatricialChannel{T,2}) where {T} = begin
	yieldMinMaxCombinations(test_operator, enumAccRepr2D(test_operator, w, r.x, r.y, size(channel)..., _ReprFake), channel, 2)
end
WExtremeModal(test_operator::Union{_TestOpGeq,_TestOpLeq}, w::Interval2D, r::_IA2DRel{R1,R2} where {R1<:_IA2DRelMaximizer,R2<:_IA2DRelMinimizer}, channel::MatricialChannel{T,2}) where {T} = begin
	yieldMinMaxCombination(test_operator, enumAccRepr2D(test_operator, w, r.x, r.y, size(channel)..., _ReprFake), channel, 2)
end

# TODO: per _TestOpLeq gli operatori si invertono

################################################################################
# END IA2D relations
################################################################################
