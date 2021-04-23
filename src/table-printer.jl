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

function extract_model(
		file_name::String,
		type::String;
		n_trees::Number = 100,
		keep_header = true,
		column_separator = ";",
		exclude_variance = true,
		exclude_parameters = [ "K", "oob_error" ],
		secondary_file_name::Union{Nothing,String} = nothing,
		remove_duplicated_rows = true
	)

	if ! isfile(file_name)
		error("No file with name $(file_name) found.")
	end

	file = open(file_name, "r")
	secondary_table =
		if isnothing(secondary_file_name)
			nothing
		else
			extract_model(
				secondary_file_name, type,
				n_trees = n_trees,
				keep_header = false,
				column_separator = column_separator,
				exclude_variance = exclude_variance,
				exclude_parameters = exclude_parameters,
				secondary_file_name = nothing
			)
		end

	function split_line(line)
		return split(chomp(line), column_separator, keepempty = true)
	end

	function get_proper_columns_indexes(header, type, n_trees)
		function get_tree_number_from_header(header, index)
			return parse(Int, split(replace(header[index], "RF(" => ""), ",")[1])
		end

		function is_variance_column(header, index)::Bool
			return contains(header[index], "σ")
		end

		function is_excluded_column(header, index)::Bool
			for excluded in exclude_parameters
				if contains(split(header[index], ")")[2], excluded)
					return true
				end
			end
			return false
		end

		local selected_columns = []
		if type == "T"
			for i in 1:length(header)
				if startswith(header[i], "T")
					if is_excluded_column(header, i)
						continue
					end
					push!(selected_columns, i)
				end
			end
		elseif type == "RF"
			for i in 1:length(header)
				if startswith(header[i], "RF") && get_tree_number_from_header(header, i) == n_trees
					if exclude_variance && is_variance_column(header, i)
						continue
					end
					if is_excluded_column(header, i)
						continue
					end
					push!(selected_columns, i)
				end
			end	
		else
			error("No model known of type $(type); could be \"T\" or \"RF\".")
		end

		selected_columns
	end
	
	header = []
	append!(header, split_line(readline(file)))

	selected_columns = [1]
	append!(selected_columns, get_proper_columns_indexes(header, type, n_trees))

	table::Vector{Vector{Any}} = []
	if keep_header
		push!(table, header[selected_columns])
	end
	for l in readlines(file)
		push!(table, split_line(l)[selected_columns])
	end

	close(file)

	if !isnothing(secondary_table)
		if remove_duplicated_rows
			# TODO: is there a less naive solution to this?
			for st_row in secondary_table
				jump_row = false
				for pt_row in table
					if st_row[1] == pt_row[1]
						jump_row = true
					end
				end
				if !jump_row
					push!(table, st_row)
				end
			end
		else
			append!(table, secondary_table)
		end
	end

	table
end

function string_table_csv(table::Vector{Vector{Any}}; column_separator = ";")
	result = ""
	for row in table
		for (i, cell) in enumerate(row)
			result *= string(cell)
			if i != length(row)
				result *= ";"
			end
		end
		result *= "\n"
	end
	result
end

function string_table_latex(table::Vector{Vector{Any}};
		column_separator = ";",
		header_size = 1,
		first_column_size = 1,
		v_lin_every_cloumn = 0,
		foot_note = "",
		scale = 1.0
	)
	result = "\\begin{table}[t]\n"

	if scale != 1.0
		result *= "\\resizebox{$(scale)\\linewidth}{!}{\\begin{minipage}{\\textwidth}"
	end

	result *= "\\begin{tabular}{$("c"^first_column_size)|"
	if v_lin_every_cloumn == 0
		result *= "$("l"^(length(table[header_size+1])-first_column_size))"
	else
		for i in (first_column_size+1):length(table[header_size+1])
			result *= "l"
			if (i-first_column_size) % v_lin_every_cloumn == 0 && i != length(table[header_size+1])
				result *= "|"
			end
		end
	end
	result *= "}\n"

	for (i, row) in enumerate(table)
		if i == 1
#			result *= "\\toprule\n"
		end
		for (j, cell) in enumerate(row)
			result *= string(cell)
			if j != length(row)
				result *= " & "
			end
		end
		if i != length(table)
			result *= " \\\\"
		end
		result *= "\n"
		if i == header_size
			result *= "\\hline"
		end
		if i == length(table)
#			result *= "\\bottomrule\n"
		end
	end
	result *= "\\end{tabular}"
	if length(foot_note) > 0
		result *= "\\begin{tablenotes}\n"
		result *= "\\item " * foot_note * "\n"
	  	result *= "\\end{tablenotes}\n"
	end

	if scale != 1.0
		result *= "\\end{minipage}}"
	end

	result *= "\n\\end{table}\n"

	result
end
