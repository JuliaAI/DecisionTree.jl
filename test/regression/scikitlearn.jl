
srand(2)
n,m = 10^3, 5 ;
features = rand(n,m);
weights = rand(-1:1,m);
labels = features * weights;

model = fit!(DecisionTreeRegressor(max_labels=5, pruning_purity_threshold=0.1), features, labels)
@test R2(labels, predict(model, features)) > 0.8

model = fit!(RandomForestRegressor(ntrees=10, max_labels=5, nsubfeatures=2), features, labels)
@test R2(labels, predict(model, features)) > 0.8

srand(2)
N = 3000
X = randn(N, 10)
y = randn(N)
maxdepth = 5
model = fit!(DecisionTreeRegressor(max_depth=maxdepth), X, y)
@test depth(model) == maxdepth


## Test that the RNG arguments work as expected
srand(2)
X = randn(100, 10)
y = randn(100)
@test fit_predict!(RandomForestRegressor(; rng=10), X, y) ==
    fit_predict!(RandomForestRegressor(; rng=10), X, y)
@test fit_predict!(RandomForestRegressor(; rng=10), X, y) !=
    fit_predict!(RandomForestRegressor(; rng=22), X, y)

