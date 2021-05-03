
import JSON

function init_new_execution_progress_dictionary(file_path::String, exec_ranges::Vector, params_names::Vector{String})
	execution_dictionaries = []

	for combination_tuple in IterTools.product(exec_ranges...)
		d = zip([params_names..., "seeds"], [combination_tuple..., []]) |> Dict
		push!(execution_dictionaries, d)
	end

	export_execution_progress_dictionary(file_path, execution_dictionaries)

	execution_dictionaries = import_execution_progress_dictionary(file_path)

	return execution_dictionaries
end

function export_execution_progress_dictionary(file_path::String, dicts::AbstractVector)
	mkpath(dirname(file_path))
	file = open(file_path, "w+")
	write(file, JSON.json(dicts))
	close(file)
end

function import_execution_progress_dictionary(file_path::String)
	file = open(file_path)
	dicts = JSON.parse(file)
	close(file)
	dicts
end

function append_in_file(file_name::String, text::String)
	mkpath(dirname(file_name))
	file = open(file_name, "a+")
	write(file, text)
	close(file)
end

function is_same_kwargs(dk1::Dict, dk2::NamedTuple{T, N}) where {T, N}
	return dk1 == Dict{String, Any}([String(k) => v for (k,v) in zip(keys(dk2),values(dk2))])
end

function is_same_kwargs(dk1::NamedTuple{T, N}, dk2::Dict) where {T, N}
	return is_same_kwargs(dk2, dk1)
end

function is_same_kwargs(dk1::NamedTuple{T, N}, dk2::NamedTuple{T, N}) where {T, N}
	return
		Dict{String, Any}([String(k) => v for (k,v) in zip(keys(dk1),values(dk1))])
		  ==
		Dict{String, Any}([String(k) => v for (k,v) in zip(keys(dk2),values(dk2))])
end

# TODO: this should be overloaded in some way; this is a really poor way to handle type specific behaviour
function _are_the_same(obj1::Any, obj2::Any)::Bool
	if obj1 isa NamedTuple || obj2 isa NamedTuple
		return is_same_kwargs(obj1, obj2)
	else
		return isequal(obj1, obj2)
	end
end

function is_combination_already_computed(exec_dicts, exec_ranges_names, params_combination, seed)::Bool
	done = false
	for dict in exec_dicts
		# if there are as many trues in left array as the number or exec_ranges_names then the iteration corresponds
		if length(findall([_are_the_same(dict[par_name], params_combination[i]) for (i, par_name) in enumerate(exec_ranges_names)])) == length(exec_ranges_names) &&
				seed in dict["seeds"]
			done = true
			break
		end
	end
	return done
end

function push_seed_to_history!(exec_dicts, exec_ranges_names, params_combination, seed)
	for dict in exec_dicts
		if length(findall([_are_the_same(dict[par_name], params_combination[i]) for (i, par_name) in enumerate(exec_ranges_names)])) == length(exec_ranges_names)
			push!(dict["seeds"], seed)
			break
		end
	end
end

function _match_filter(test_parameters, filters)::Bool
	for filter in filters
		for (i, k) in enumerate(keys(filter))
			# TODO: handle test_parameters has no key "k"
			if filter[k] == test_parameters[k]
				if i == length(filter)
					# if was it was the last key then there is a match
					return true
				else
					# if key has same value continue cycling through filter keys
					continue
				end
			else
				# if there is a key not matching go to next filter
				break
			end
		end
	end

	return false
end

# note: filters may contain less keys than test_parameters
function is_whitelisted_test(test_parameters, filters = [])::Bool
	# if filters is empty no whitelisting is applied
	if length(filters) == 0
		return true
	end

	return _match_filter(test_parameters, filters)
end

function is_blacklisted_test(test_parameters, filters = [])::Bool
	# if filters is empty no blacklisting is applied
	if length(filters) == 0
		return false
	end

	return _match_filter(test_parameters, filters)
end

function load_or_create_execution_progress_dictionary(file_path::String, args...)
	if isfile(file_path)
		import_execution_progress_dictionary(file_path)
	else
		init_new_execution_progress_dictionary(file_path, args...)
	end
end
