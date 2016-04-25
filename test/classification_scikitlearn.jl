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
@test mean(predict(model, features) .== labels) > 0.9

model = fit!(RandomForestClassifier(), features, labels)
@test mean(predict(model, features) .== labels) > 0.9

model = fit!(AdaBoostStumpClassifier(), features, labels)
# Adaboost isn't so hot on this task
@test mean(predict(model, features) .== labels) > 0.6
