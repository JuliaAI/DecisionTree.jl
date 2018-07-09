using ScikitLearnBase
using DecisionTree
using Test

import Distributed
import Random

function mean(xs)
    return sum(xs) / length(xs)
end

function run_tests(list)
    for test in list
        println("TEST: $test \n")
        include(test)
        println("=" ^ 50)
    end
end

classification = [
    "classification/random.jl",
    "classification/heterogeneous.jl",
    "classification/digits.jl",
    "classification/scikitlearn.jl"
]

regression =     [
    "regression/random.jl",
    "regression/digits.jl",
    "regression/scikitlearn.jl"
]

miscellaneous =  [
    "miscellaneous/promote.jl",
    "miscellaneous/parallel.jl"
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

