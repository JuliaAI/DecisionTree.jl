julia

Pkg.activate("DecisionTree.jl")
using Revise
using Pkg

import Random
# Random.GLOBAL_RNG
my_rng = Random.MersenneTwister(1)

using Logging

using BenchmarkTools
using DecisionTree
using DecisionTree.ModalLogic

include("example-datasets.jl")


# n_samp = 200
# N = 50
# (X,Y) = simpleDataset(n_samp,N)
# (X_train,Y_train,X_test,Y_test) = traintestsplit(X,Y,0.8)

# (X_train,Y_train,X_test,Y_test) = EduardDataset(5)
(X_train,Y_train,X_test,Y_test) = EduardDataset(10)


# XX = ModalLogic.OntologicalDataset{Int64,1}(ModalLogic.IntervalOntology,X_train)


# global_logger(ConsoleLogger(stderr, Logging.Warn))
global_logger(ConsoleLogger(stderr, Logging.Info))

# Timings history
# -- Add the use of Sf
# 133.009 ms (2253001 allocations: 194.44 MiB)
# -- Switched from WorldSet to WorldVector
# 131.344 ms (2251791 allocations: 194.32 MiB)
# Well, the difference is subtle
# 130.677 ms (2251632 allocations: 194.31 MiB)
# -- Actually, I've made a mistake (I was updating Sf[i] insfead of S[...])
# 133.435 ms (2251632 allocations: 194.31 MiB)
# @btime T2 = build_tree(Y_train, X_train; ontology = ModalLogic.IntervalOntology, rng = my_rng)

T2 = build_tree(Y_train, X_train; rng = my_rng)

preds = apply_tree(T2, X_test)

if Y_test == preds
	print("100% Accuracy baby!")
else
	print("Accuracy: ", sum(Y_test .== preds)/length(preds))
end
