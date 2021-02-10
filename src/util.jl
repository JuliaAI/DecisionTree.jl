
# written by Poom Chiarawongse <eight1911@gmail.com>

module util

	export Label, gini, entropy, zero_one, q_bi_sort!

	const Label = Int
	
	# This function translates a list of labels into categorical form
	function assign(Y :: AbstractVector{T}) where T
		function assign(Y :: AbstractVector{T}, list :: AbstractVector{T}) where T
			dict = Dict{T, Label}()
			@simd for i in 1:length(list)
				@inbounds dict[list[i]] = i
			end

			_Y = Array{Label}(undef, length(Y))
			@simd for i in 1:length(Y)
				@inbounds _Y[i] = dict[Y[i]]
			end

			return list, _Y
		end

		set = Set{T}()
		for y in Y
			push!(set, y)
		end
		list = collect(set)
		return assign(Y, list)
	end

	@inline function zero_one(ns :: AbstractVector{T}, n :: T) where {T <: Real}
		return 1.0 - maximum(ns) / n
	end

	@inline function gini(ns :: AbstractVector{T}, n :: T) where {T <: Real}
		s = 0.0
		@simd for k in ns
			s += k * (n - k)
		end
		return s / (n * n)
	end

	# returns the entropy of ns/n, ns is an array of integers
	# and entropy_terms are precomputed entropy terms
	@inline function entropy(ns::AbstractVector{U}, n, entropy_terms) where {U <: Integer}
		s = 0.0
		for k in ns
			s += entropy_terms[k+1]
		end
		return log(n) - s / n
	end

	@inline function entropy(ns :: AbstractVector{T}, n :: T) where {T <: Real}
		s = 0.0
		@simd for k in ns
			if k > 0
				s += k * log(k)
			end
		end
		return log(n) - s / n
	end

	# adapted from the Julia Base.Sort Library
	@inline function partition!(v::AbstractVector, w::AbstractVector{T}, pivot::T, region::Union{AbstractVector{<:Integer},UnitRange{<:Integer}}) where T
		i, j = 1, length(region)
		r_start = region.start - 1
		@inbounds while true
			while i <= length(region) && w[i] <= pivot; i += 1; end; # TODO check this i <= ... sign
			while j >= 1              && w[j]  > pivot; j -= 1; end;
			i >= j && break
			ri = r_start + i
			rj = r_start + j
			v[ri], v[rj] = v[rj], v[ri]
			w[i], w[j] = w[j], w[i]
			i += 1; j -= 1
		end
		return j
	end

	# adapted from the Julia Base.Sort Library
	function insert_sort!(v::AbstractVector, w::AbstractVector, lo::Integer, hi::Integer, offset::Integer)
		@inbounds for i = lo+1:hi
			j = i
			x = v[i]
			y = w[offset+i]
			while j > lo
				if x < v[j-1]
					v[j] = v[j-1]
					w[offset+j] = w[offset+j-1]
					j -= 1
					continue
				end
				break
			end
			v[j] = x
			w[offset+j] = y
		end
		return v
	end

	@inline function _selectpivot!(v::AbstractVector, w::AbstractVector, lo::Integer, hi::Integer, offset::Integer)
		@inbounds begin
			mi = (lo+hi)>>>1

			# sort the values in v[lo], v[mi], v[hi]

			if v[mi] < v[lo]
				v[mi], v[lo] = v[lo], v[mi]
				w[offset+mi], w[offset+lo] = w[offset+lo], w[offset+mi]
			end
			if v[hi] < v[mi]
				if v[hi] < v[lo]
					v[lo], v[mi], v[hi] = v[hi], v[lo], v[mi]
					w[offset+lo], w[offset+mi], w[offset+hi] = w[offset+hi], w[offset+lo], w[offset+mi]
				else
					v[hi], v[mi] = v[mi], v[hi]
					w[offset+hi], w[offset+mi] = w[offset+mi], w[offset+hi]
				end
			end

			# move v[mi] to v[lo] and use it as the pivot
			v[lo], v[mi] = v[mi], v[lo]
			w[offset+lo], w[offset+mi] = w[offset+mi], w[offset+lo]
			v_piv = v[lo]
			w_piv = w[offset+lo]
		end

		# return the pivot
		return v_piv, w_piv
	end

	# adapted from the Julia Base.Sort Library
	@inline function _bi_partition!(v::AbstractVector, w::AbstractVector, lo::Integer, hi::Integer, offset::Integer)
		pivot, w_piv = _selectpivot!(v, w, lo, hi, offset)
		# pivot == v[lo], v[hi] > pivot
		i, j = lo, hi
		@inbounds while true
			i += 1; j -= 1
			while v[i] < pivot; i += 1; end;
			while pivot < v[j]; j -= 1; end;
			i >= j && break
			v[i], v[j] = v[j], v[i]
			w[offset+i], w[offset+j] = w[offset+j], w[offset+i]
		end
		v[j], v[lo] = pivot, v[j]
		w[offset+j], w[offset+lo] = w_piv, w[offset+j]

		# v[j] == pivot
		# v[k] >= pivot for k > j
		# v[i] <= pivot for i < j
		return j
	end


	# adapted from the Julia Base.Sort Library
	# adapted from the Julia Base.Sort Library
	# this sorts v[lo:hi] and w[offset+lo, offset+hi]
	# simultaneously by the values in v[lo:hi]
	const SMALL_THRESHOLD  = 20
	function q_bi_sort!(v::AbstractVector, w::AbstractVector, lo::Integer, hi::Integer, offset::Integer)
		@inbounds while lo < hi
			hi-lo <= SMALL_THRESHOLD && return insert_sort!(v, w, lo, hi, offset)
			j = _bi_partition!(v, w, lo, hi, offset)
			if j-lo < hi-j
				# recurse on the smaller chunk
				# this is necessary to preserve O(log(n))
				# stack space in the worst case (rather than O(n))
				lo < (j-1) && q_bi_sort!(v, w, lo, j-1, offset)
				lo = j+1
			else
				j+1 < hi && q_bi_sort!(v, w, j+1, hi, offset)
				hi = j-1
			end
		end
		return v
	end

end

