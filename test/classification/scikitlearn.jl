@testset "scikitlearn.jl" begin

Random.seed!(2)
n,m = 10^3, 5 ;
features = rand(n,m);
weights = rand(-1:1,m);
labels = round.(Int, features * weights);

# I wish we could use ScikitLearn.jl's cross-validation, but that'd require
# installing it on Travis
model = fit!(DecisionTreeClassifier(pruning_purity_threshold=0.9), features, labels)
@test mean(predict(model, features) .== labels) > 0.8
@test feature_importances(model) == feature_importances(model.root)
p1 = permutation_importances(model, features, labels)
p2 = permutation_importances(model, features, labels)
@test all(@. (p1.mean - p2.mean) / sqrt((p1.std ^ 2 + p2.std ^2)/2) < 3)
@test findall(>(0.1), dropcol_importances(model, features, labels).mean) == [2, 5]

model = fit!(RandomForestClassifier(), features, labels)
@test mean(predict(model, features) .== labels) > 0.8
@test feature_importances(model) == feature_importances(model.ensemble)
p1 = permutation_importances(model, features, labels)
p2 = permutation_importances(model, features, labels)
@test all(@. (p1.mean - p2.mean) / sqrt((p1.std ^ 2 + p2.std ^2)/2) < 3)
@test findall(>(0.1), dropcol_importances(model, features, labels).mean) == [2, 5]

model = fit!(AdaBoostStumpClassifier(), features, labels)
# Adaboost isn't so hot on this task, disabled for now
mean(predict(model, features) .== labels)
@test feature_importances(model) == feature_importances((model.ensemble, model.coeffs))
p1 = permutation_importances(model, features, labels)
p2 = permutation_importances(model, features, labels)
@test all(filter(!isnan, @. (p1.mean - p2.mean) / sqrt((p1.std ^ 2 + p2.std ^2)/2)) .< 3)
@test argmax(dropcol_importances(model, features, labels).mean) in [2, 5]

Random.seed!(2)
N = 3000
X = randn(N, 10)
# TODO: we should probably support fit!(::DecisionTreeClassifier, ::BitArray)
y = convert(Vector{Bool}, randn(N) .< 0)
max_depth = 5
model = fit!(DecisionTreeClassifier(max_depth=max_depth), X, y)
@test depth(model) == max_depth


## Test that the RNG arguments work as expected
Random.seed!(2)
X = randn(100, 10)
y = rand(Bool, 100);

@test predict_proba(fit!(RandomForestClassifier(; rng=10), X, y), X) ==
    predict_proba(fit!(RandomForestClassifier(; rng=10), X, y), X)
@test predict_proba(fit!(RandomForestClassifier(; rng=10), X, y), X) !=
    predict_proba(fit!(RandomForestClassifier(; rng=12), X, y), X)

end # @testset
