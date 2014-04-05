using DecisionTree

n,m = 10^3, 5 ;
features = randn(n,m);
weights = rand(-2:2,m);
labels = features * weights;

# regression tree
model = build_tree(labels, features)
apply_tree(model, [5.9,3.0,5.1,1.9,0.0])
nfoldCV_tree(labels, features, 3)

# regression forest
model = build_forest(labels,features, 3, 10)
apply_forest(model, [5.9,3.0,5.1,1.9,0.0])
nfoldCV_forest(labels, features, 2, 10, 3)

