
include("table-printer.jl")

import Statistics
#tables_path = "./tables"
#mkpath(tables_path)

seeds = [
    "7933197233428195239",
    "1735640862149943821",
    "3245434972983169324",
    #"2461790151364820568", TODO: add this if encase finishes in time
    "1708661255722239460",
    "1107158645723204584"
]

seed_dictionary = Dict{String,String}(zip(seeds, "s" * string(i) for i in 1:length(seeds)))

secondary_file_name = nothing
if length(ARGS) > 1
    secondary_file_name = ARGS[2]
end

destination = "./"
if length(ARGS) > 2
    destination = ARGS[3]
end

T = extract_model(ARGS[1], "T", secondary_file_name = secondary_file_name)
RF50 = extract_model(ARGS[1], "RF", n_trees = 50, secondary_file_name = secondary_file_name)
RF100 = extract_model(ARGS[1], "RF", n_trees = 100, secondary_file_name = secondary_file_name)

function get_task(row_label)
    return split(row_label, ",", limit = 2)[2]
end

function get_seed(row_label)
    return split(row_label, ",", limit = 2)[1]
end

function get_model_type(h)
    return split(h, "(", limit = 2)[1]
end
function get_column_var(h)
    return split(h, ")", limit = 2)[2]
end
function get_n_trees(h)
    return split(split(h, "(", limit = 2)[2], ",")[1]
end
function get_model(h)
    return lstrip(split(split(h, get_n_trees(h), limit = 2)[2], ")")[1], ',')
end

function table_to_dict(model)
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

function dict_to_desired_format_table(dict, header, seed_order, task_order; average_row_decorate = "", use_only_seeds = [], beautify_latex = false)
    average_string = "avg(std)"
    function average_row(dict)::Vector{Any}
        new_row::Vector{Any} = [average_row_decorate * average_string]
        rows = collect(values(dict))
        for i in 2:length(rows[1])
            # TODO: there is a better way to do this, I know it
            vals::Vector{Real} = []
            for r in rows
                push!(vals, parse(Float64, r[i]))
            end
            push!(new_row, string(round(Statistics.mean(vals), digits=2), " (", round(Statistics.std(vals), digits=2), ")"))
        end
        new_row
    end

    function pretty_header(header)::Vector{Vector{Any}}
        type = get_model_type(header[2])

        models::Vector{String} = []
        for i in 2:length(header)
            mod = get_model(header[i])
            if !(mod in models)
                push!(models, mod)
            end
        end
        vars::Vector{String} = []
        for i in 2:length(header)
            if get_model(header[i]) == models[1]
                push!(vars, get_column_var(header[i]))
            end
        end

        nh::Vector{Vector{Any}} = []
        if type == "RF"
            # first row
            new_row_1::Vector{Any} = [ "\\multirow{2}{*}{$(type) ($(get_n_trees(header[2])))}" ]
            for (j, m) in enumerate(models)
                res = "\\multicolumn{$(length(vars))}{c"
                if j != length(models)
                    res *= "|"
                end
                res *= "}{$(m)}"
                push!(new_row_1,  res)
            end
            push!(nh, new_row_1)
            # second row
            new_row_2::Vector{Any} = [ "" ]
            for i in 1:length(models)
                for v in vars
                    push!(new_row_2, v)
                end
            end
            push!(nh, new_row_2)
        elseif type == "T"
            new_row::Vector{Any} = [ "$(type)" ]
            for i in 1:length(models)
                for v in vars
                    push!(new_row, v)
                end
            end
            push!(nh, new_row)
        end

        nh
    end

    function prettify_first_column(table, header_size)::Vector{Vector{Any}}
        new_table::Vector{Vector{Any}} = []
        for i in 1:header_size
            if i > 1
                splice!(table[i], 1:0, [""])
            end
            push!(new_table, table[i])
        end
        new_table[1][1] = "\\multicolumn{2}{c|}{$(new_table[1][1])}"

        task_already_inserted::Vector{Any} = []

        for i in (header_size+1):length(table)
            if startswith(table[i][1], average_row_decorate * average_string)
                table[i][1] = average_row_decorate * "\\multicolumn{2}{c|}{$(replace(table[i][1], average_row_decorate => ""))}"
            else
                seed = get_seed(table[i][1])
                task = replace(replace(get_task(table[i][1]), "1," => "", count = 1), ", " => ",")
                task = replace(task, "(30," => "(")
                task = replace(task, "1," => "C,", count = 1)
                task = replace(task, "2," => "B,", count = 1)
                table[i][1] = seed_dictionary[seed]
                if !(task in task_already_inserted)
                    k = i
                    row_count = 1
                    while startswith(table[k][1], average_row_decorate * average_string)
                        k += 1
                        row_count += 1
                    end
                    splice!(table[i], 1:0, ["\\multirow{$(row_count)}{*}{\\begin{sideways}$(task)\\end{sideways}}"])
                    push!(task_already_inserted, task)
                else
                    splice!(table[i], 1:0, [""])
                end
            end
            push!(new_table, table[i])
        end

        new_table
    end

    table::Vector{Vector{Any}} = []
    new_header::Vector{Vector{Any}} = []
    if beautify_latex
        append!(new_header, pretty_header(header))
    else
        append!(new_header, [header])
    end
    header_size = length(new_header)
    append!(table, new_header)
    for task in task_order
        if !haskey(dict, task)
            continue
        end
        for seed in seed_order
            if length(use_only_seeds) != 0 && !(seed in use_only_seeds)
                continue
            end
            if !haskey(dict[task], seed)
                continue
            end
            push!(table, dict[task][seed])
        end
        push!(table, average_row(dict[task]))
    end

    row_size = length(table[header_size+1])
    
    new_t::Vector{Vector{Any}} = []
    if beautify_latex
        new_t = prettify_first_column(table, header_size)
    else
        new_t = table
    end

    first_column_size = (length(table[header_size+1])-row_size) + 1

    new_t, header_size, first_column_size
end

# Create dicts
Ttuple = table_to_dict(T)
RF50tuple = table_to_dict(RF50)
RF100tuple = table_to_dict(RF100)

# Arrange table in proper way (CSV)
Ttable_csv = dict_to_desired_format_table(Ttuple..., use_only_seeds = seeds, beautify_latex = false)
RF50table_csv = dict_to_desired_format_table(RF50tuple..., use_only_seeds = seeds, beautify_latex = false)
RF100table_csv = dict_to_desired_format_table(RF100tuple..., use_only_seeds = seeds, beautify_latex = false)

color_row = ""#"\\rowcolor{lightgray!50}"
# Arrange table in proper way (LaTeX)
Ttable_latex = dict_to_desired_format_table(Ttuple...; average_row_decorate = color_row, beautify_latex = true, use_only_seeds = seeds)
RF50table_latex = dict_to_desired_format_table(RF50tuple...; average_row_decorate = color_row, beautify_latex = true, use_only_seeds = seeds)
RF100table_latex = dict_to_desired_format_table(RF100tuple...; average_row_decorate = color_row, beautify_latex = true, use_only_seeds = seeds)

# Prepare string to print (CSV)
csv_doc = string_table_csv(Ttable_csv[1]) * "\n"
csv_doc *= string_table_csv(RF50table_csv[1]) * "\n"
csv_doc *= string_table_csv(RF100table_csv[1]) * "\n"

# Prepare string to print (LaTeX)
seed_item = "Seeds: "

for (i, (k, v)) in enumerate(seed_dictionary)
    global seed_item *= k * " = " * v
    seed_item *= i == length(seed_dictionary) ? "." : ", "
end

latex_t = string_table_latex(Ttable_latex[1], header_size=Ttable_latex[2], first_column_size = Ttable_latex[3], v_lin_every_cloumn = 4, scale = 0.45) * "\n"
latex_rf50 = string_table_latex(RF50table_latex[1], header_size=RF50table_latex[2], first_column_size = RF50table_latex[3], v_lin_every_cloumn = 4, scale = 0.45) * "\n"
latex_rf100 = string_table_latex(RF100table_latex[1], header_size=RF100table_latex[2], first_column_size = RF100table_latex[3], v_lin_every_cloumn = 4, scale = 0.45) * "\n"

function write_to_file(file_name, string::String)
    file = open(file_name, "w")

    write(file, string)

    close(file)
end

write_to_file(destination * "/tree.tex", latex_t)
write_to_file(destination * "/rf50.tex", latex_rf50)
write_to_file(destination * "/rf100.tex", latex_rf100)

nice_tree = "\\begin{lstlisting}
⟨⟩ (V20 ⫺₇₀ 0.5615922788407576)
✔ ⟨L̅⟩ (V7 ⫹₇₀ 6.176777086476566e-7)
│✔ 1 : 6/6
│✘ ⟨E̅⟩ (V54 ⫺₇₀ 6.064393361449956)
│ ✔ 1 : 4/4
│ ✘ V14 ⫹₇₀ 0.820070925146093
│  ✔ ⟨A̅⟩ (V14 ⫺₇₀ 0.35871902278168466)
│  │✔ ⟨L⟩ (V24 ⫺₇₀ 0.05022759203477401)
│  ││✔ 1 : 9/9
│  ││✘ 0 : 2/2
│  │✘ 0 : 24/25
│  ✘ 0 : 26/26
✘ ⟨⟩ (V45 ⫺₇₀ 0.0005385868666157022)
 ✔ ⟨A̅⟩ (V51 ⫺₇₀ 0.06956334660096865)
 │✔ ⟨A̅⟩ (V46 ⫺₇₀ 0.32768843936211767)
 ││✔ ⟨B̅⟩ (V10 ⫺₇₀ 0.44057750825408043)
 │││✔ 1 : 6/6
 │││✘ 0 : 9/10
 ││✘ 1 : 53/55
 │✘ ⟨E⟩ (V20 ⫺₇₀ 0.029182827872923228)
 │ ✔ ⟨D⟩ (V48 ⫹₇₀ 4.199406045996452e-6)
 │ │✔ 1 : 7/7
 │ │✘ ⟨E̅⟩ (V5 ⫹₇₀ 0.07069683637970851)
 │ │ ✔ ⟨E̅⟩ (V8 ⫹₇₀ 0.16965271511272867)
 │ │ │✔ ⟨A⟩ (V24 ⫺₇₀ 0.07407156769668966)
 │ │ ││✔ 1 : 3/3
 │ │ ││✘ 0 : 11/11
 │ │ │✘ 1 : 8/8
 │ │ ✘ 0 : 15/15
 │ ✘ 1 : 12/13
 ✘ 0 : 22/24
\\end{lstlisting}"

# Print CSV
#println(csv_doc)

# Print LaTeX
#println(latex_doc)
