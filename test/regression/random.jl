srand(5)

n,m = 10^3, 5 ;
features = randn(n,m);
weights = rand(-2:2,m);
labels = features * weights;

# over-fitting
max_features = 1
model = build_tree(labels, features, max_features)
preds = apply_tree(model, features)
@test R2(labels, preds) > 0.99      # R2: coeff of determination

max_features = 5
max_depth = 3
nsubfeatures = 0                    # all features
model = build_tree(labels, features, max_features, nsubfeatures, max_depth)
@test depth(model) == max_depth

println("\n##### nfoldCV Regression Tree #####")
r2 = nfoldCV_tree(labels, features, 3)
@test mean(r2) > 0.6

println("\n##### nfoldCV Regression Forest #####")
r2 = nfoldCV_forest(labels, features, 2, 10, 3)
@test mean(r2) > 0.8
