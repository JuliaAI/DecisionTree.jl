# One unique world (propositional case)
struct OneWorld    <: AbstractWorld
	OneWorld() = new()
	OneWorld(w::_emptyWorld) = new()
	OneWorld(w::_firstWorld) = new()
	OneWorld(w::_centeredWorld) = new()
end;

show(io::IO, w::OneWorld) = begin
	print(io, "−")
end

worldTypeDimensionality(::Type{OneWorld}) = 0
print_world(::OneWorld) = println("−")
