function loaddata()
    f = open("data/digits.csv")
    data = readlines(f)[2:end]
    data = [[parse(Float32, i) 
        for i in split(row, ",")]
        for row in data]
    data = hcat(data...)
    Y = Int.(data[1, 1:end]) + 1
    X = convert(Matrix, transpose(data[2:end, 1:end]))

    return X, Y

end

num_leaves(node::DecisionTree.Node) = num_leaves(node.left) + num_leaves(node.right)
num_leaves(node::DecisionTree.Leaf) = 1

X, Y = loaddata()

# CLASSIFICATION CHECK
t = DecisionTree.build_tree(Y, X)
@test num_leaves(t) == 148

t = DecisionTree.build_tree(Y, X, 0, 6)
@test num_leaves(t) == 57

t = DecisionTree.build_tree(Y, X, 0, 6, 5)
@test num_leaves(t) == 50

t = DecisionTree.build_tree(Y, X, 0, 6, 3, 5)
@test num_leaves(t) == 55

t = DecisionTree.build_tree(Y, X, 0, 6, 3, 5, 0.05)
@test num_leaves(t) == 54


# REGRESSION CHECK
Y += 0.0 # convert X and Y to Float to enable regression
X += 0.0

t = DecisionTree.build_tree(Y, X)
@test num_leaves(t) in [190, 191]

t = DecisionTree.build_tree(Y, X, 5)
@test num_leaves(t) in [190, 191]

t = DecisionTree.build_tree(Y, X, 5, 0, 6)
@test num_leaves(t) == 50

t = DecisionTree.build_tree(Y, X, 5, 0, 6, 5)
@test num_leaves(t) == 44

t = DecisionTree.build_tree(Y, X, 5, 0, 6, 3, 5)
@test num_leaves(t) == 45

t = DecisionTree.build_tree(Y, X, 5, 0, 6, 3, 5, 0.05)
@test num_leaves(t) == 44
