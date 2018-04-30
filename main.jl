include("src/DecisionTree.jl")

function loaddata(large=false)
    data = if large
        f = open("./mnist_train.csv")
        data = readlines(f)[1:2000]
        [map(parse, split(row, ",")) for row in data]
    else
        f = open("./digits.csv")
        data = readlines(f)[2:end]
        [[parse(Float32, i) 
            for i in split(row, ",")]
            for row in data]
    end

    data = hcat(data...)
    Y = Int.(data[1, 1:end]) + 1
    X = transpose(data[2:end, 1:end])

    return X, Y

end

function depth(node)
    if typeof(node) === DecisionTree.Leaf
        return 0
    end
    return 1 + max(depth(node.left), depth(node.right))
end

num_leaves(node::DecisionTree.Node) = 1 + num_leaves(node.left) + num_leaves(node.right)
num_leaves(node::DecisionTree.Leaf) = 1

X, Y = loaddata()
# X += rand(size(X)...) / 3
println("data")
@time DecisionTree.build_tree(Y, X)
@time DecisionTree.build_tree(Y, X)
@time DecisionTree.build_tree(Y, X)
@time node = DecisionTree.build_tree(Y, X)
@time for i in 1:100
    DecisionTree.build_tree(Y, X)
end
print(num_leaves(node))