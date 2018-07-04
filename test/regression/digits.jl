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

Y = float.(Y) # labels/targets to Float to enable regression

t = DecisionTree.build_tree(Y, X, 1)
@test length(t) in [190, 191]

t = DecisionTree.build_tree(Y, X, 5)
@test length(t) == 126

t = DecisionTree.build_tree(Y, X, 5, 0, 6)
@test length(t) == 44
@test depth(t) == 6

t = DecisionTree.build_tree(Y, X, 1, 0, -1, 20)
@test length(t) == 122

t = DecisionTree.build_tree(Y, X, 1, 0, -1, 2, 0.25)
@test length(t) == 103

end # @testset
