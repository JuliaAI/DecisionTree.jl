@testset "digits.jl" begin

X, Y = load_data("digits")

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
        min_purity_increase,
        rng=StableRNG(1))
@test length(t) == 54
i1 = impurity_importance(t, normalize = true)
s1 = split_importance(t)
p1 = permutation_importance(t, Y, X, (model, y, X)->accuracy(y, apply_tree(model, X)), rng=StableRNG(1)).mean

# test that all purity decisions are based on passed-in purity function;
# if so, this should be same as previous test
entropy1000(ns, n) = DecisionTree.util.entropy(ns, n) * 1000
min_samples_leaf    = 3
min_samples_split   = 5
min_purity_increase = 0.05 * 1000
t = DecisionTree.build_tree(
        Y, X,
        n_subfeatures, max_depth,
        min_samples_leaf,
        min_samples_split,
        min_purity_increase,
        loss = entropy1000,
        rng=StableRNG(1))
@test length(t) == 54
i2 = impurity_importance(t, normalize = true)
s2 = split_importance(t)
p2 = permutation_importance(t, Y, X, (model, y, X)->accuracy(y, apply_tree(model, X)), rng=StableRNG(1)).mean
@test isapprox(i2, i1)
@test s1 == s2
@test similarity(p2, p1) > 0.9

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
        min_purity_increase;
        rng=StableRNG(1))
preds = apply_forest(model, X)
cm = confusion_matrix(Y, preds)
@test cm.accuracy > 0.95

preds_MT = apply_forest(model, X, use_multithreading = true)
cm_MT = confusion_matrix(Y, preds_MT)
@test cm_MT.accuracy > 0.95

n_iterations        = 100
model, coeffs = DecisionTree.build_adaboost_stumps(
        Y, X,
        n_iterations);
preds = apply_adaboost_stumps(model, coeffs, X);
cm = confusion_matrix(Y, preds)
@test cm.accuracy > 0.8

end # @testset
