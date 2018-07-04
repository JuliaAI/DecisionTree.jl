@testset "digits.jl" begin

function loaddata()
    f = open("data/digits.csv")
    data = readlines(f)[2:end]
    data = [[parse(Float32, i)
        for i in split(row, ",")]
        for row in data]
    data = hcat(data...)
    Y = Int.(data[1, 1:end]) .+ 1
    X = convert(Matrix, transpose(data[2:end, 1:end]))
    return X, Y
end

X, Y = loaddata()

t = DecisionTree.build_tree(Y, X)
@test length(t) == 148
@test sum(apply_tree(t, X) .== Y) == length(Y)

t = DecisionTree.build_tree(Y, X, 0, 6)
@test length(t) == 57

t = DecisionTree.build_tree(Y, X, 0, 6, 5)
@test length(t) == 50

t = DecisionTree.build_tree(Y, X, 0, 6, 3, 5)
@test length(t) == 55

t = DecisionTree.build_tree(Y, X, 0, 6, 3, 5, 0.05)
@test length(t) == 54

end # @testset
