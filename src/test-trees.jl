include("test-header.jl")

using FileIO

trees_directory = "./results-audio-scan/trees-to-test"
output_directory = "./results-test-trees"

mkpath(trees_directory)
mkpath(output_directory)

# tree_files = map(str -> trees_directory * "/" * str, filter!(endswith(".jld"), readdir(trees_directory)))

# function load_tree(file_name::String)::Union{DecisionTree.DTree,DecisionTree.DTInternal}
# 	println("Loading $(file_name)...")
# 	load(string(file_name))["T"]
# end

# function load_all_trees(file_names::Vector{String})::Vector{DecisionTree.DTree,DecisionTree.DTInternal}
# 	arr = []
# 	for f in file_names
# 		push!(arr, load_tree(f))
# 	end
# 	arr
# end

# testing_tree = load_tree(tree_files[1])
# print_tree(testing_tree)

# seeds = [
# 	"7933197233428195239",
# 	"1735640862149943821",
# 	"3245434972983169324",
# 	"1708661255722239460",
# 	"1107158645723204584"
# ]

# configs = [
# 	"30.45.30",
# 	"30.75.50"
# ]

# tree_401a21a62325fa736d2c57381d6b6a1f63a92e665220ac3c63ccca89efe7fb0c
# tree_b7269f8be54c95a26094482fcd29b38d9611ddc2fcd9a14015cbdf4fab67aa28
# tree_72ce352289cb551233bc542db93dd90ab643b82e182e9b1a0d7fd532c479b4e2
# tree_014838ab30d9d70e2d5939ee4da07c21e07fe69960cbfdffe0348ecd6f167c71

# for seed in seeds
#     for config in configs

# for (dataset_str,tree_filepath) in [
# 	("1708661255722239460.1.1.40.(30.25.15)","tree_401a21a62325fa736d2c57381d6b6a1f63a92e665220ac3c63ccca89efe7fb0c"),
# 	("1708661255722239460.1.2.40.(30.45.30)","tree_b7269f8be54c95a26094482fcd29b38d9611ddc2fcd9a14015cbdf4fab67aa28"),
# 	("1708661255722239460.1.2.20.(30.75.50)","tree_72ce352289cb551233bc542db93dd90ab643b82e182e9b1a0d7fd532c479b4e2"),
# 	("1107158645723204584.1.2.20.(30.75.50)","tree_014838ab30d9d70e2d5939ee4da07c21e07fe69960cbfdffe0348ecd6f167c71")]

obj = Dict{String, String}("7933197233428195239,1,1,40,(30.25.15)" => [
"tree_e77119ed524a04463ce65ceacf7e2fe3fd01d4c18c813f38d0043872a014fc9e",
], "7933197233428195239,1,1,60,(30.75.50)" => [
"tree_4a3930a9b121b248b6e1ee1ac8fb6630b3954fe9b3c6a6660869667173aac46d",
], "7933197233428195239,1,1,60,(30.45.30)" => [
"tree_f8623be8bd56a2fd20b8738fecd14f30f13feec38aee4c44970eeaebf0f6342c",
], "7933197233428195239,1,2,20,(30.75.50)" => [
"tree_9378d387d2aa5088ddabab153600764a49d3e5be7a39f77c78b1e7bf062025cb",
], "7933197233428195239,1,2,20,(30.45.30)" => [
"tree_91a9b442a9086eb15502d6c4d578109b82ba881eaf5a34335508555739242e03",
], "7933197233428195239,1,2,40,(30.75.50)" => [
"tree_a97c981d761492ec5f6b21459f06735282952cb6f6b7e13ccd77a208b7da97d4",
], "7933197233428195239,1,2,40,(30.45.30)" => [
"tree_6da775ba99031f8497918e5ea9f5ffd95209c0273e2afffc0e2a7b54379f9db8",
], "1735640862149943821,1,1,40,(30.25.15)" => [
"tree_72eb426a99f160bbfebaf6759134358428fea9b45179f77dc9c08495981953b0",
], "1735640862149943821,1,1,60,(30.75.50)" => [
"tree_2ff9f28e8bcbbb7ce631001aff9e64dd433c7952be070e58d995e6327fcfff5c",
], "1735640862149943821,1,1,60,(30.45.30)" => [
"tree_e70336028f2ce0a657b4a2fc61e94415e7c71cd320204f597bb772aa68b1dc1e",
], "1735640862149943821,1,2,20,(30.75.50)" => [
"tree_b9c4a396a3f10dcf8a15dbb7c0fed5c09d4ea1ff13f172ae45ae85e25d938290",
], "1735640862149943821,1,2,20,(30.45.30)" => [
"tree_8c404d7d63fa8b694fe3d09ff522aa2bc6db53e76b69226ae00cc7af894e8521",
], "1735640862149943821,1,2,40,(30.75.50)" => [
"tree_5114fa8989c787e80135f52eabd71b30a273198869bb1ebec4f51456fb35604e",
], "1735640862149943821,1,2,40,(30.45.30)" => [
"tree_5c8a9668a5b362778a0ca9c06e1e48d67980ca2f1432743c690338ce2dc3da66",
], "3245434972983169324,1,1,40,(30.25.15)" => [
"tree_d53619b5016071cb2d6f8c1252e792b0894c8f8fad75ae54f587ed6bbfee73b5",
], "3245434972983169324,1,1,60,(30.75.50)" => [
"tree_0e8d200572ba2ddf822ec3c4c9561b12cc9dc18043a5aeeba32fe51397e1a921",
], "3245434972983169324,1,1,60,(30.45.30)" => [
"tree_df295094c227ab6410fd0fde5d8553801be0fe15666a250cc6d53f576e18d9a5",
], "3245434972983169324,1,2,20,(30.75.50)" => [
"tree_79f6965de3e40fe720562fbd89ab974d2df593fce19e83e4f07d22ff9345178c",
], "3245434972983169324,1,2,20,(30.45.30)" => [
"tree_72503035564eae1b102cff316b3d16d53a83279d3022c90e9705cc4ae3ff705e",
], "3245434972983169324,1,2,40,(30.75.50)" => [
"tree_81f01e9ce81c1ef417e976da67bbb55c9125db094065c2d43b4b50482c06ba0b",
], "3245434972983169324,1,2,40,(30.45.30)" => [
"tree_53b9847b2e2d07a2a39a10a99fd188c7834cef95788bf06d4d63df0531ed721d",
], "1708661255722239460,1,1,40,(30.25.15)" => [
"tree_401a21a62325fa736d2c57381d6b6a1f63a92e665220ac3c63ccca89efe7fb0c",
], "1708661255722239460,1,1,60,(30.75.50)" => [
"tree_6d6e580fa0e0ade00a88e2d7bff1d5216747ac63a82ccdbf43a537a5126e3153",
], "1708661255722239460,1,1,60,(30.45.30)" => [
"tree_dc50592ad7f619bb3de264721ca4efe8d9b9054abd766e7290e44f84685f792d",
], "1708661255722239460,1,2,20,(30.75.50)" => [
"tree_72ce352289cb551233bc542db93dd90ab643b82e182e9b1a0d7fd532c479b4e2",
], "1708661255722239460,1,2,20,(30.45.30)" => [
"tree_c65e834abd6fbbc69215ca9dcdef37823c879b36ea89b35a6fae95cb54dd398b",
], "1708661255722239460,1,2,40,(30.75.50)" => [
"tree_655b1b3e9b624011944a968f27062a9c89e05e4cc1aab05c38e6d099db22392e",
], "1708661255722239460,1,2,40,(30.45.30)" => [
"tree_b7269f8be54c95a26094482fcd29b38d9611ddc2fcd9a14015cbdf4fab67aa28",
], "1107158645723204584,1,1,40,(30.25.15)" => [
"tree_73e0842816f2f44199401f63e7896abfb4837031e9b5452ef652db26b10a09d0",
], "1107158645723204584,1,1,60,(30.75.50)" => [
"tree_542c05767aded3641f2f8c7111e2eb5870f82522f68f739c88ea6e79910803ad",
], "1107158645723204584,1,1,60,(30.45.30)" => [
"tree_826b411f4a2a25d923f2315f20e8d3e0b8005a25b529e0c78dc5d9b0f873fe7f",
], "1107158645723204584,1,2,20,(30.75.50)" => [
"tree_014838ab30d9d70e2d5939ee4da07c21e07fe69960cbfdffe0348ecd6f167c71",
], "1107158645723204584,1,2,20,(30.45.30)" => [
"tree_107d6153efb7ae3c2c3d1f0ee5575ae5f521bd180350c09196ecfbb386aba268",
], "1107158645723204584,1,2,40,(30.75.50)" => [
"tree_863d23bdd06a9cf772ec48da1248dbdbe1944a0ecefffdd1f442a10b9a9691b4",
], "1107158645723204584,1,2,40,(30.45.30)" => [
"tree_4bd2db23b606e4052264dac4ff316632e0dba236e7807440791fbb25bc1b982e",
])
for (dataset_str,tree_filepaths) in obj
for tree_filepath in tree_filepaths
	dataset = nothing
	tree = nothing
	JLD2.@load "./results-audio-scan/datasets/$(dataset_str)-balanced-test.jld" dataset
	tree = load("./results-audio-scan/trees/$(tree_filepath).jld")["T"]
	# JLD2.@load "./results-audio-scan/datasets/$(seed).1.1.60.($(config)).jld" dataset

	X, Y = dataset[1], dataset[2]
	# Y = 1 .- Y

	# print_tree(tree, n_tot_inst = 226)
	print_tree(tree)
	regenerated_tree = print_apply_tree(tree, X, Y)
	readline()
	# print_tree(regenerated_tree)
end
end

#     end
# end
