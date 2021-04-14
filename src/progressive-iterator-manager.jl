
import JSON

function init_new_execution_progress_dictionary(file_path::String, exec_n_tasks, exec_n_versions, exec_nbands, exec_dataset_kwargs)
	execution_dictionaries = []

	for n_task in exec_n_tasks
		for n_version in exec_n_versions
			for nbands in exec_nbands
				for dataset_kwargs in exec_dataset_kwargs
					push!(execution_dictionaries, Dict(
						"n_task" => n_task,
						"n_version" => n_version,
						"nbands" => nbands,
						"dataset_kwargs" => dataset_kwargs,
						"runs" => []
					))
				end
			end
		end
	end

	export_execution_progress_dictionary(file_path, execution_dictionaries)

	execution_dictionaries = import_execution_progress_dictionary(file_path)

	return execution_dictionaries
end

function export_execution_progress_dictionary(file_path::String, dicts::AbstractVector)
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

function load_or_create_execution_progress_dictionary(file_path::String, exec_n_tasks, exec_n_versions, exec_nbands, exec_dataset_kwargs)
	if isfile(file_path)
		import_execution_progress_dictionary(file_path)
	else
		init_new_execution_progress_dictionary(file_path, exec_n_tasks, exec_n_versions, exec_nbands, exec_dataset_kwargs)
	end
end
