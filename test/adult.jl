using Base.Test
using DecisionTree

run(`wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data`);
adult = readcsv("adult.data");

features = adult[:, 1:14];
labels = adult[:, 15];

println("\n##### 3 foldCV Classification Tree #####")
accuracy = nfoldCV_tree(labels, features, 0.9, 3);
@test mean(accuracy) > 0.8

println("\n##### 3 foldCV Classification Forest #####")
accuracy = nfoldCV_forest(labels, features, 2, 10, 3, 0.5);
@test mean(accuracy) > 0.8
