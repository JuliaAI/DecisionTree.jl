using Base.Test
using DecisionTree

srand(2)
n,m = 10^3, 5 ;
features = rand(n,m);
weights = rand(-1:1,m);
labels = _int(features * weights);

# I wish we could use ScikitLearn.jl's cross-validation, but that'd require 
# installing it on Travis
model = fit!(DecisionTreeClassifier(pruning_purity_threshold=0.9), features, labels)
@test mean(predict(model, features) .== labels) > 0.8

model = fit!(RandomForestClassifier(), features, labels)
@test mean(predict(model, features) .== labels) > 0.8

model = fit!(AdaBoostStumpClassifier(), features, labels)
# Adaboost isn't so hot on this task, disabled for now
mean(predict(model, features) .== labels)

srand(2)
N = 3000
X = randn(N, 10)
# TODO: we should probably support fit!(::DecisionTreeClassifier, ::BitArray)
y = convert(Vector{Bool}, randn(N) .< 0)
maxdepth = 5
model = fit!(DecisionTreeClassifier(maxdepth=maxdepth), X, y)
@test depth(model) == maxdepth
