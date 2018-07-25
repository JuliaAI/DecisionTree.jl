# Regression Test - Appliances Energy Prediction Data Set
# https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction

@testset "energy.jl" begin

download("https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv", "energy.csv");
energy = readcsv("energy.csv");

features = energy[2:end, 3:end];
labels = float.(energy[2:end, 2]);

# over-fitting
n_subfeatures       = 0
max_depth           = -1
min_samples_leaf    = 1
model = build_tree(
        labels, features,
        n_subfeatures,
        max_depth,
        min_samples_leaf)
preds = apply_tree(model, features);
@test R2(labels, preds) > 0.99

println("\n##### nfoldCV Regression Tree #####")
r2 = nfoldCV_tree(labels, features, 3);
@test mean(r2) > 0.05

println("\n##### nfoldCV Regression Forest #####")
r2 = nfoldCV_forest(labels, features, 2, 10, 3);
@test mean(r2) > 0.35

end # @testset
