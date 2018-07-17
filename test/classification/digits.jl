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


n_subfeatures       = 0
max_depth           = 6
min_samples_leaf    = 5
t = DecisionTree.build_tree(
        Y, X,
        n_subfeatures, max_depth)
@test length(t) == 57

t = DecisionTree.build_tree(
        Y, X,
        n_subfeatures, max_depth,
        min_samples_leaf)
@test length(t) == 50

min_samples_leaf    = 3
min_samples_split   = 5
min_purity_increase = 0.05
t = DecisionTree.build_tree(
        Y, X,
        n_subfeatures, max_depth,
        min_samples_leaf,
        min_samples_split)
@test length(t) == 55

t = DecisionTree.build_tree(
        Y, X,
        n_subfeatures, max_depth,
        min_samples_leaf,
        min_samples_split,
        min_purity_increase)
@test length(t) == 54

n_subfeatures       = 3
n_trees             = 10
partial_sampling    = 0.7
max_depth           = -1
min_samples_leaf    = 1
min_samples_split   = 2
min_purity_increase = 0.0
model = DecisionTree.build_forest(
        Y, X,
        n_subfeatures,
        n_trees,
        partial_sampling,
        max_depth,
        min_samples_leaf,
        min_samples_split,
        min_purity_increase)
preds = apply_forest(model, X)
cm = confusion_matrix(Y, preds)
@test cm.accuracy > 0.95

n_iterations        = 100
model, coeffs = DecisionTree.build_adaboost_stumps(
        Y, X,
        n_iterations);
preds = apply_adaboost_stumps(model, coeffs, X);
cm = confusion_matrix(Y, preds)
@test cm.accuracy > 0.8

end # @testset
