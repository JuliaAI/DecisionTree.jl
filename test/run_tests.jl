using Base.Test
using DecisionTree

tests = ["iris.jl", "regression_rand.jl"]

println("Running tests...")
for curtest in tests
    println("Test: $curtest")
    include(curtest)
    println("=" ^ 50)
end
