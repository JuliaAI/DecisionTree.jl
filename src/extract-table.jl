
include("table-printer.jl")

import Statistics
#tables_path = "./tables"
#mkpath(tables_path)

secondary_file_name = nothing
if length(ARGS) > 1
    secondary_file_name = ARGS[2]
end

T = extract_model(ARGS[1], "T", secondary_file_name = secondary_file_name)
#RF50 = extract_model(ARGS[1], "RF", n_trees = 50, secondary_file_name = secondary_file_name)
#RF100 = extract_model(ARGS[1], "RF", n_trees = 100, secondary_file_name = secondary_file_name)

function table_to_dict(model)
    function get_task(row_label)
        return split(row_label, ",", limit = 2)[2]
    end

    function get_seed(row_label)
        return split(row_label, ",", limit = 2)[1]
    end

    seed_order = Vector{String}()
    task_order = Vector{String}()

    for (i, row) in enumerate(model)
        if i == 1
            continue
        end
        seed = get_seed(row[1])
        if !(seed in seed_order)
            push!(seed_order, seed)
        end
        task = get_task(row[1])
        if !(task in task_order)
            push!(task_order, task)
        end
    end

    dict = Dict{String,Dict{String,AbstractVector}}()
    for (i, row) in enumerate(model)
        if i == 1
            continue
        end
        task = get_task(row[1])
        seed = get_seed(row[1])

        if !haskey(dict, task)
            dict[task] = Dict{String,AbstractVector}()
        end

        dict[task][seed] = model[i]
    end

    dict, model[1], seed_order, task_order
end
#tasks

function dict_to_desired_format_table(dict, header, seed_order, task_order)
    function average_row(dict)::Vector{Any}
        new_row::Vector{Any} = ["average (std)"]
        rows = collect(values(dict))
        for i in 2:length(rows[1])
            # TODO: there is a better way to do this, I know it
            vals::Vector{Real} = []
            for r in rows
                push!(vals, parse(Float64, r[i]))
            end
            push!(new_row, string(Statistics.mean(vals), " (", Statistics.std(vals), ")"))
        end
        new_row
    end

    table::Vector{Vector{Any}} = []
    push!(table, header)
    for task in task_order
        if !haskey(dict, task)
            continue
        end
        for seed in seed_order
            if !haskey(dict[task], seed)
                continue
            end
            push!(table, dict[task][seed])
        end
        push!(table, average_row(dict[task]))
    end

    table
end

Tdict, header, seed_order, task_order = table_to_dict(T)

table = dict_to_desired_format_table(Tdict, header, seed_order, task_order)

println(string_table_csv(table))

#print(string_table_csv(T))
#println("===================================================================")
#print(string_table_csv(RF50))
#println("===================================================================")
#print(string_table_csv(RF100))

#    "7933197233428195239"
#    "1735640862149943821"
#    "3245434972983169324"
#    "1708661255722239460"
#    "1107158645723204584"
