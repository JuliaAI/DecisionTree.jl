using DecisionTree

if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end


classification = ["./classification/random.jl",
                  "./classification/heterogeneous.jl",
                  "./classification/digits.jl",
                  "./classification/scikitlearn.jl"]

regression = ["./regression/random.jl",
              "./regression/scikitlearn.jl"]

miscellaneous = ["./miscellaneous/promote.jl"]


function run_tests(list)
    for test in list
        println("TEST: $test")
        include(test)
        println("=" ^ 50)
    end
end

for list in [classification, regression, miscellaneous]
    run_tests(list)
end
