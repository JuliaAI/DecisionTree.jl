@testset "scikitlearn.jl" begin

n, m = 10^3, 5;
features = rand(StableRNG(1), n, m);
weights = rand(StableRNG(1), -1:1, m);
labels = features * weights;

let
    regressor = DecisionTreeRegressor(; rng=StableRNG(1), min_samples_leaf=5, pruning_purity_threshold=0.1)
    model = fit!(regressor, features, labels)
    @test R2(labels, predict(model, features)) > 0.8
end

model = fit!(DecisionTreeRegressor(; rng=StableRNG(1), min_samples_split=5), features, labels)
@test R2(labels, predict(model, features)) > 0.8
@test feature_importances(model) == feature_importances(model.root)
p1 = permutation_importances(model, features, labels)
p2 = permutation_importances(model, features, labels)
@test all(@. (p1.mean - p2.mean) / sqrt((p1.std ^ 2 + p2.std ^2)/2) < 3)
@test findall(>(0.1), dropcol_importances(model, features, labels).mean) == [2, 5]

let
    regressor = RandomForestRegressor(; rng=StableRNG(1), n_trees=10, min_samples_leaf=5, n_subfeatures=2)
    model = fit!(regressor, features, labels)
    @test R2(labels, predict(model, features)) > 0.8
end

N = 3000
X = randn(StableRNG(1), N, 10)
y = randn(StableRNG(1), N)
max_depth = 5
model = fit!(DecisionTreeRegressor(; rng=StableRNG(1), max_depth=max_depth), X, y)
@test depth(model) == max_depth


## Test that the RNG arguments work as expected
X = randn(StableRNG(1), 100, 10)
y = randn(StableRNG(1), 100)
@test fit_predict!(RandomForestRegressor(; rng=10), X, y) ==
    fit_predict!(RandomForestRegressor(; rng=10), X, y)
@test fit_predict!(RandomForestRegressor(; rng=10), X, y) !=
    fit_predict!(RandomForestRegressor(; rng=22), X, y)

end # @testset
