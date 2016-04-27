using Base.Test
using DecisionTree

srand(2)
n,m = 10^3, 5 ;
features = rand(n,m);
weights = rand(-1:1,m);
labels = features * weights;

model = fit!(DecisionTreeRegressor(maxlabels=5, pruning_purity_threshold=0.1), features, labels)
@test R2(labels, predict(model, features)) > 0.8

model = fit!(RandomForestRegressor(ntrees=10, maxlabels=5, nsubfeatures=2), features, labels)
@test R2(labels, predict(model, features)) > 0.8
