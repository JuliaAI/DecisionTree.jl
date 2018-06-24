### Promote Leaf to Node

@testset "promote.jl" begin

leaf = Leaf(0, [0])
node = Node(1, 1, leaf, leaf)

[leaf, node]
[node, leaf]

end # @testset
