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

model = fit!(RandomForestClassifier(), features, labels)
@test mean(predict(model, features) .== labels) > 0.8

model = fit!(AdaBoostStumpClassifier(), features, labels)
# Adaboost isn't so hot on this task, disabled for now
mean(predict(model, features) .== labels)

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
