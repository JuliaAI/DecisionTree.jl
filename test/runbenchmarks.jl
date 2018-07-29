using DecisionTree
using BenchmarkTools

import Random
import DelimitedFiles

include("benchmark/classification_suite.jl")
include("benchmark/regression_suite.jl")

Random.srand(1)

function load_digits()
    f = open("data/digits.csv")
    data = readlines(f)[2:end]
    data = [[parse(Float32, i)
        for i in split(row, ",")]
        for row in data]
    data = hcat(data...)
    Y = Int.(data[1, 1:end]) .+ 1
    X = convert(Matrix, transpose(data[2:end, 1:end]))
    return X, Y
end

function load_adult()
    adult = DelimitedFiles.readdlm("data/adult.csv", ',');
    n = floor(Int, size(adult)[1] / 10)
    X = adult[1:n, 1:14];
    Y = adult[1:n, 15];
    return X, Y    
end

function load_iris()
    iris = DelimitedFiles.readdlm("data/iris.csv", ',')
    X = iris[:, 1:4]
    Y = iris[:, 5]
    return X, Y    
end

function pad(s::String, l::Int=21)
    t = length(s)
    p = max(0, l - t)
    return s * " " ^ p
end

function print_details(results)
    k = keys(results)
    for i in k
        s = "================ " * i * " ================"
        println("\n" * s)
        display(results[i])
#        if typeof(results[i]) <: BenchmarkGroup
#            print_details(results[i])
#        end
    end
end


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
