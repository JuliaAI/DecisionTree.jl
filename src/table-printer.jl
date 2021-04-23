###############################################################################
############################ OUTPUT HANDLERS ##################################
###############################################################################

id_f = x->x
half_f = x->ceil(Int, x/2)
sqrt_f = x->ceil(Int, sqrt(x))

function print_function(func::Core.Function)::String
	if func === id_f
		"all"
	elseif func === half_f
		"half"
	elseif func === sqrt_f
		"sqrt"
	elseif func === DecisionTree.util.entropy
		"entropy"
	else
		""
	end
end

function print_tree_head(io, tree_args)
	write(io, "T($(print_function(tree_args.loss)),$(tree_args.min_samples_leaf),$(tree_args.min_purity_increase),$(tree_args.min_loss_at_leaf))")
end

function print_forest_head(io, forest_args)
	write(io, "RF($(forest_args.n_trees),$(print_function(forest_args.n_subfeatures)),$(print_function(forest_args.n_subrelations)))")
end

function print_head(
		io::Core.IO,
		tree_args::AbstractArray,
		forest_args::AbstractArray;
		separator = ";",
		tree_columns = ["K", "sensitivity", "specificity", "precision", "accuracy"],
		forest_columns = ["K", "σ² K", "sensitivity", "σ² sensitivity", "specificity", "σ² specificity", "precision", "σ² precision", "accuracy", "σ² accuracy", "oob_error", "σ² oob_error"],
        empty_column_before = 1
	) # where {T, N}

	for i in 1:empty_column_before
		write(io, separator)
	end

	for i in 1:length(tree_args)
		for j in 1:length(tree_columns)
			print_tree_head(io, tree_args[i])
			write(io, tree_columns[j])
			# if !(i === length(tree_args) && j === length(tree_columns)) || length(forest_args) > 0
			write(io, separator)
			# end
		end
	end

	for i in 1:length(forest_args)
		for j in 1:length(forest_columns)
			print_forest_head(io, forest_args[i])
			write(io, forest_columns[j])
			# if !(i === length(forest_args) && j === length(forest_columns))
			write(io, separator)
			# end
		end
	end
	write(io, "\n")
end

function print_head(tree_args::NamedTuple{T,N}, forest_args::AbstractArray; kwargs...) where {T, N}
	print_head(stdout, tree_args, forest_args; kwargs...)
end

###############################################################################
###############################################################################
