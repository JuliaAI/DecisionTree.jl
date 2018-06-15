download("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", "adult.csv");
adult = readcsv("adult.csv");

features = adult[:, 1:14];
labels = adult[:, 15];

model = build_tree(labels, features)
preds = apply_tree(model, features)
cm = confusion_matrix(labels, preds)
@test cm.accuracy > 0.99

println("\n##### 3 foldCV Classification Tree #####")
accuracy = nfoldCV_tree(labels, features, 0.9, 3);
@test mean(accuracy) > 0.8

println("\n##### 3 foldCV Classification Forest #####")
accuracy = nfoldCV_forest(labels, features, 2, 10, 3, 0.5);
@test mean(accuracy) > 0.8
