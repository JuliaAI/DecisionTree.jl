@testset "scikitlearn.jl" begin

Random.seed!(2)
n,m = 10^3, 5 ;
features = rand(n,m);
weights = rand(-1:1,m);
labels = features * weights;

model = fit!(DecisionTreeRegressor(min_samples_leaf=5, pruning_purity_threshold=0.1), features, labels)
@test R2(labels, predict(model, features)) > 0.8
@test impurity_importance(model) == impurity_importance(model.root)
@test isapprox(permutation_importance(model, features, labels, rng = 1).mean, permutation_importance(model.root, labels, features, (model, y, X) -> R2(y, apply_tree(model, X)), rng = 1).mean)

model = fit!(DecisionTreeRegressor(min_samples_split=5), features, labels)
@test R2(labels, predict(model, features)) > 0.8
@test split_importance(model) == split_importance(model.root)
@test isapprox(permutation_importance(model, features, labels, rng = 1).mean, permutation_importance(model.root, labels, features, (model, y, X) -> R2(y, apply_tree(model, X)), rng = 1).mean)

model = fit!(RandomForestRegressor(n_trees=10, min_samples_leaf=5, n_subfeatures=2), features, labels)
@test R2(labels, predict(model, features)) > 0.8
@test impurity_importance(model) == impurity_importance(model.ensemble)
@test split_importance(model) == split_importance(model.ensemble)
@test isapprox(permutation_importance(model, features, labels, rng = 1).mean, permutation_importance(model.ensemble, labels, features, (model, y, X) -> R2(y, apply_forest(model, X)), rng = 1).mean)

Random.seed!(2)
N = 3000
X = randn(N, 10)
y = randn(N)
max_depth = 5
model = fit!(DecisionTreeRegressor(max_depth=max_depth), X, y)
@test depth(model) == max_depth

## Test that the RNG arguments work as expected
Random.seed!(2)
X = randn(100, 10)
y = randn(100)
@test fit_predict!(RandomForestRegressor(; rng=10), X, y) ==
    fit_predict!(RandomForestRegressor(; rng=10), X, y)
@test fit_predict!(RandomForestRegressor(; rng=10), X, y) !=
    fit_predict!(RandomForestRegressor(; rng=22), X, y)

end # @testset
