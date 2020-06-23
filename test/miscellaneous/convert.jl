# Test conversion of Leaf to Node

@testset "convert.jl" begin

lf = Leaf(1, [1])
nv = Node{Int, Int}[]
push!(nv, lf)
@test apply_tree(nv[1], [0]) == 1

lf = Leaf(1.0, [0.0, 1.0])
nv = Node{Int, Float64}[]
push!(nv, lf)
@test apply_tree(nv[1], [0]) == 1.0

lf = Leaf("A", ["B", "A"])
nv = Node{Int, String}[]
push!(nv, lf)
@test apply_tree(nv[1], [0]) == "A"

end # @testset
