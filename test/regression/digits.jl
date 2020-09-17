@testset "digits.jl" begin

X, Y = load_data("digits")

Y = float.(Y) # labels/targets to Float to enable regression

min_samples_leaf    = 1
n_subfeatures       = 0
max_depth           = -1
t = DecisionTree.build_tree(
        Y, X,
        n_subfeatures,
        max_depth,
        min_samples_leaf)
@test length(t) in [190, 191]

min_samples_leaf    = 5
t = DecisionTree.build_tree(
        Y, X,
        n_subfeatures,
        max_depth,
        min_samples_leaf)
@test length(t) == 126

min_samples_leaf    = 5
n_subfeatures       = 0
max_depth           = 6
t = DecisionTree.build_tree(
        Y, X,
        n_subfeatures,
        max_depth,
        min_samples_leaf)
@test length(t) == 44
@test depth(t) == 6

min_samples_leaf    = 1
n_subfeatures       = 0
max_depth           = -1
min_samples_split   = 20
t = DecisionTree.build_tree(
        Y, X,
        n_subfeatures,
        max_depth,
        min_samples_leaf,
        min_samples_split)
@test length(t) == 122

min_samples_leaf    = 1
n_subfeatures       = 0
max_depth           = -1
min_samples_split   = 2
min_purity_increase = 0.25
t = DecisionTree.build_tree(
        Y, X,
        n_subfeatures,
        max_depth,
        min_samples_leaf,
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

println("\n##### 3 foldCV Regression Tree #####")
n_folds = 5
r2 = nfoldCV_tree(Y, X, n_folds; verbose=false);
@test mean(r2) > 0.6

println("\n##### 3 foldCV Regression Forest #####")
n_subfeatures = 2
n_trees = 10
n_folds = 5
partial_sampling = 0.5
r2 = nfoldCV_forest(Y, X, n_folds, n_subfeatures, n_trees, partial_sampling; verbose=false)
@test mean(r2) > 0.6

end # @testset
