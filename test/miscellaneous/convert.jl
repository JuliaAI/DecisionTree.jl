# Test conversion of Leaf to Node

@testset "convert.jl" begin

lf = Leaf(1, [1])

nv = Node{Int, Int}[]
push!(nv, lf)

@test apply_tree(nv[1], [0]) == 1

end # @testset
