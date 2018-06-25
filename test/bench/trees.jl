

include("../../src/DecisionTree.jl")

function loaddata()
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

num_leaves(node::DecisionTree.Node) = num_leaves(node.left) + num_leaves(node.right)
num_leaves(node::DecisionTree.Leaf) = 1

X, Y = loaddata()

# for compilation
for i in 1:10
    t = DecisionTree.build_tree(Y, X)
end

println("[ === CLASSIFICATION BENCHMARK === ]")
for j in 1:3
    @time for i in 1:100
        tree = DecisionTree.build_tree(Y, X)
    end
end


Y = float.(Y) # labels/targets to Float to enable regression

# for compilation
for i in 1:10
    t = DecisionTree.build_tree(Y, X)
end

println("[ === REGRESSION BENCHMARK === ]")
for j in 1:3
    @time for i in 1:100
        tree = DecisionTree.build_tree(Y, X)
    end
end