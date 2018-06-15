# Test parallelization of random forests

addprocs()
@test nprocs() > 1

@everywhere using DecisionTree

srand(16)

# Classification
n,m = 10^3, 5;
features = rand(n,m);
weights = rand(-1:1,m);
labels = _int(features * weights);

model = build_forest(labels, features, 2, 10);
preds = apply_forest(model, features);
cm = confusion_matrix(labels, preds);
@test cm.accuracy > 0.9


# Regression
n,m = 10^3, 5 ;
features = randn(n,m);
weights = rand(-2:2,m);
labels = features * weights;

model = build_forest(labels, features, 2, 10);
preds = apply_forest(model, features);
@test R2(labels, preds) > 0.9
