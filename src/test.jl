julia

using Pkg

using Revise
Pkg.activate("DecisionTree.jl")

# using StaticArrays
# X = ModalLogic.OntologicalDataset{Int64,0}(ModalLogic.IntervalAlgebra,Array{Int64,2}(undef, 20, 10))
# X = ModalLogic.OntologicalDataset{Float32,0}(ModalLogic.IntervalAlgebra,rand(Float32, 20, 10))

# TODO generate simple dataset with 0s and 1s, two classes and try different starting entity. Indeed, we can parametrize, the initial entity.

# Xdom = Array{Int,3}(undef, 10, 1, 3)
# Y = []
# for i in 1:10
# 	instance = fill(0, 3)
# 	y = rand(0:1)
# 	if y == 0
# 		instance[3] += 1
# 	else
# 		instance[1] += 1
# 	end
# 	Xdom[i,1,:] .= instance
# 	push!(Y,y)
# end


using Revise
# using BenchmarkTools

a = 3
x = 5
N = 15

AllXdom = Array{Int,3}(undef, N, 1, 3);
AllY = Array{Int,1}(undef, N);
for i in 1:N
	instance = fill(2, 3)
	y = rand(0:1)
	if y == 0
		instance[3] = 1
	else
		instance[3] = 2
	end
	AllXdom[i,1,:] .= instance
	AllY[i] = y
end

AllXdom[1,1,:]

spl = floor(Int, N*.8)
Xdom = AllXdom[1:spl,:,:]
Y    = AllY[1:spl]
Xdom_test = AllXdom[spl+1:end,:,:]
Y_test    = AllY[spl+1:end]

using Logging
logger = SimpleLogger(stdout, Logging.Debug)
old_logger = global_logger(logger)

# include("DecisionTree.jl/src/DecisionTree.jl")
# using OhMyREPL
import Random
using BenchmarkTools
using DecisionTree
using DecisionTree.ModalLogic

global_logger(ConsoleLogger(stderr, Logging.Warn))

@btime T2 = DecisionTree.build_tree(Y,Xdom) # 132.899 μs (1687 allocations: 96.00 KiB)


# @btime T = DecisionTree.treeclassifier.fit(
# 	X = OntologicalDataset{Int,1}(IntervalAlgebra,Xdom),
# 	Y = Y,
# 	W = nothing,
# 	loss = DecisionTree.util.entropy,
# 	max_features = 1,
# 	max_depth = 2,
# 	min_samples_leaf = 1,
# 	min_samples_split = 2,
# 	min_purity_increase = 0.0,
# 	rng = Random.GLOBAL_RNG)
# T2 = DecisionTree._convert(T.root, T.list, Y[T.labels])
# DecisionTree.print_tree(T2)

# T = DecisionTree.build_tree(Y,Xdom)
# DecisionTree.print_tree(T)
# TODO
# apply_tree(Xdom_test, Y_test
