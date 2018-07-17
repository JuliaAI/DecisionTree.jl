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

min_samples_leaf    = 1
t = DecisionTree.build_tree(
    Y, X,
    min_samples_leaf)
@test length(t) in [190, 191]

min_samples_leaf    = 5
t = DecisionTree.build_tree(
        Y, X,
        min_samples_leaf)
@test length(t) == 126

min_samples_leaf    = 5
n_subfeatures       = 0
max_depth           = 6
t = DecisionTree.build_tree(
        Y, X,
        min_samples_leaf,
        n_subfeatures,
        max_depth)
@test length(t) == 44
@test depth(t) == 6

min_samples_leaf    = 1
n_subfeatures       = 0
max_depth           = -1
min_samples_split   = 20
t = DecisionTree.build_tree(
        Y, X,
        min_samples_leaf,
        n_subfeatures,
        max_depth,
        min_samples_split)
@test length(t) == 122

min_samples_leaf    = 1
n_subfeatures       = 0
max_depth           = -1
min_samples_split   = 2
min_purity_increase = 0.25
t = DecisionTree.build_tree(
        Y, X,
        min_samples_leaf,
        n_subfeatures,
        max_depth,
        min_samples_split,
        min_purity_increase)
@test length(t) == 103


n_subfeatures       = 3
n_trees             = 10
partial_sampling    = 0.7
max_depth           = -1
min_samples_leaf    = 5
min_samples_split   = 2
min_purity_increase = 0.0
model = build_forest(
        Y, X,
        n_subfeatures,
        n_trees,
        partial_sampling,
        max_depth,
        min_samples_leaf,
        min_samples_split,
        min_purity_increase)
preds = apply_forest(model, X)
@test R2(Y, preds) > 0.8

end # @testset
