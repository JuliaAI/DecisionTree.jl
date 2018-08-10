using DecisionTree
using BenchmarkTools
using Random

include("benchmark/classification_suite.jl")
include("benchmark/regression_suite.jl")
include("benchmark/utils.jl")

Random.seed!(1)


# Classification Benchmarks
classification_tree = benchmark_classification(build_tree, apply_tree)
println("\n\n############### CLASSIFICATION: BUILD TREE ###############")
print_details(classification_tree["BUILD"])
println("\n\n############### CLASSIFICATION: APPLY TREE ###############")
print_details(classification_tree["APPLY"])

classification_forest = benchmark_classification(build_forest, apply_forest)
println("\n\n############### CLASSIFICATION: BUILD FOREST ###############")
print_details(classification_forest["BUILD"])
println("\n\n############### CLASSIFICATION: APPLY FOREST ###############")
print_details(classification_forest["APPLY"])

classification_adaboost = benchmark_classification(build_adaboost, apply_adaboost)
println("\n\n############### CLASSIFICATION: BUILD ADABOOST ###############")
print_details(classification_adaboost["BUILD"])
println("\n\n############### CLASSIFICATION: APPLY ADABOOST ###############")
print_details(classification_adaboost["APPLY"])



# Regression Benchmarks
regression_tree = benchmark_regression(build_tree, apply_tree)
println("\n\n############### REGRESSION: BUILD TREE ###############")
print_details(regression_tree["BUILD"])
println("\n\n############### REGRESSION: APPLY TREE ###############")
print_details(regression_tree["APPLY"])

regression_forest = benchmark_regression(build_forest, apply_forest)
println("\n\n############### REGRESSION: BUILD FOREST ###############")
print_details(regression_forest["BUILD"])
println("\n\n############### REGRESSION: APPLY FOREST ###############")
print_details(regression_forest["APPLY"])
