# Test conversion of Leaf to Node

@testset "convert.jl" begin
    lf = Leaf(1, [1])
    nv = Node{Int,Int}[]
    rv = Root{Int,Int}[]
    push!(nv, lf)
    push!(rv, lf)
    push!(rv, nv[1])
    @test apply_tree(nv[1], [0]) == 1
    @test apply_tree(rv[1], [0]) == 1
    @test apply_tree(rv[2], [0]) == 1

    lf = Leaf(1.0, [0.0, 1.0])
    nv = Node{Int,Float64}[]
    rv = Root{Int,Float64}[]
    push!(nv, lf)
    push!(rv, lf)
    push!(rv, nv[1])
    @test apply_tree(nv[1], [0]) == 1.0
    @test apply_tree(rv[1], [0]) == 1.0
    @test apply_tree(rv[2], [0]) == 1.0

    lf = Leaf("A", ["B", "A"])
    nv = Node{Int,String}[]
    rv = Root{Int,String}[]
    push!(nv, lf)
    push!(rv, lf)
    push!(rv, nv[1])
    @test apply_tree(nv[1], [0]) == "A"
    @test apply_tree(rv[1], [0]) == "A"
    @test apply_tree(rv[2], [0]) == "A"
end

@testset "convert to text" begin
    n, m = 10^3, 5
    features = rand(StableRNG(1), n, m)
    weights = rand(StableRNG(1), -1:1, m)
    labels = features * weights
    model = fit!(DecisionTreeRegressor(; rng=StableRNG(1)), features, labels)
    # Smoke test.
    print_tree(devnull, model)
end
