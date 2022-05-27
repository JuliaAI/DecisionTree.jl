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
f1 = feature_importances(t)
p1 = permutation_importances(t, Y, X, (model, y, X)->accuracy(y, apply_tree(model, X))).mean
@test similarity(f1, p1) > 0.9

t = DecisionTree.build_tree(
        Y, X,
        n_subfeatures, max_depth,
        min_samples_leaf)
@test length(t) == 50
f2 = feature_importances(t)
p2 = permutation_importances(t, Y, X, (model, y, X)->accuracy(y, apply_tree(model, X))).mean
@test similarity(f2, f1) > 0.9
@test similarity(p2, p1) > 0.9

min_samples_leaf    = 3
min_samples_split   = 5
min_purity_increase = 0.05
t = DecisionTree.build_tree(
        Y, X,
        n_subfeatures, max_depth,
        min_samples_leaf,
        min_samples_split)
@test length(t) == 55
f1 = feature_importances(t)
p1 = permutation_importances(t, Y, X, (model, y, X)->accuracy(y, apply_tree(model, X))).mean
@test similarity(f2, f1) > 0.9
@test similarity(p2, p1) > 0.9

t = DecisionTree.build_tree(
        Y, X,
        n_subfeatures, max_depth,
        min_samples_leaf,
        min_samples_split,
        min_purity_increase)
@test length(t) == 54
f1 = feature_importances(t)
p1 = permutation_importances(t, Y, X, (model, y, X)->accuracy(y, apply_tree(model, X))).mean

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
        loss = entropy1000)
@test length(t) == 54
f2 = feature_importances(t)
p2 = permutation_importances(t, Y, X, (model, y, X)->accuracy(y, apply_tree(model, X))).mean
@test similarity(f2, f1) > 0.99
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
        min_purity_increase)
preds = apply_forest(model, X)
cm = confusion_matrix(Y, preds)
@test cm.accuracy > 0.95
f1 = feature_importances(model)
p1 = permutation_importances(model, Y, X, (model, y, X)->accuracy(y, apply_forest(model, X))).mean
# Not stable
similarity(p1, f1) > 0.8

n_iterations        = 100
model, coeffs = DecisionTree.build_adaboost_stumps(
        Y, X,
        n_iterations);
preds = apply_adaboost_stumps(model, coeffs, X);
cm = confusion_matrix(Y, preds)
@test cm.accuracy > 0.8
f1 = feature_importances(model)
p1 = permutation_importances((model, coeffs), Y, X, (model, y, X)->accuracy(y, apply_adaboost_stumps(model, X))).mean
# Not stable
similarity(p1, f1) < 0.8

end # @testset
