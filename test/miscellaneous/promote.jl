### Promote Leaf to Node

@testset "promote.jl" begin

leaf = Leaf(0, [0])
node = Node(1, 1, leaf, leaf)

ln = [leaf, node]
@test length(ln) == 2

nl = [node, leaf]
@test length(nl) == 2

end # @testset
