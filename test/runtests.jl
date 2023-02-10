import AbstractTrees

using DecisionTree
using DelimitedFiles
using Random
using ScikitLearnBase
using StableRNGs
using Statistics
using Test
using LinearAlgebra

using DecisionTree: accuracy, R2, majority_vote, mean_squared_error
using DecisionTree: confusion_matrix, ConfusionMatrix

println("Julia version: ", VERSION)

similarity(a, b) = first(reshape(a, 1, :) * b / norm(a) / norm(b))

function run_tests(list)
    for test in list
        println("TEST: $test \n")
        include(test)
        println("=" ^ 50)
    end
end

classification = [
    "classification/random.jl",
    "classification/low_precision.jl",
    "classification/heterogeneous.jl",
    "classification/digits.jl",
    "classification/iris.jl",
    "classification/adult.jl",
    "classification/scikitlearn.jl"
]

regression =     [
    "regression/random.jl",
    "regression/low_precision.jl",
    "regression/digits.jl",
    "regression/scikitlearn.jl"
]

miscellaneous =  [
    "miscellaneous/convert.jl",
    "miscellaneous/abstract_trees_test.jl",
    "miscellaneous/feature_importance_test.jl",
    "miscellaneous/ensemble_methods.jl",
#    "miscellaneous/parallel.jl"

]

test_suites = [
    ("Classification", classification),
    ("Regression", regression),
    ("Miscellaneous", miscellaneous),
]

@testset "Test Suites" begin
    for ts in 1:length(test_suites)
        name = test_suites[ts][1]
        list = test_suites[ts][2]
        let
            @testset "$name" begin
                run_tests(list)
            end
        end
    end
end
