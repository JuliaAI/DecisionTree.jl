
# written by Poom Chiarawongse <eight1911@gmail.com>

module util

    export gini, entropy, zero_one, q_bi_sort!, hypergeometric

    function assign(Y :: Vector{T}, list :: Vector{T}) where T
        dict = Dict{T, Int}()
        @simd for i in 1:length(list)
            @inbounds dict[list[i]] = i
        end

        _Y = Array{Int}(undef, length(Y))
        @simd for i in 1:length(Y)
            @inbounds _Y[i] = dict[Y[i]]
        end

        return list, _Y
    end

    function assign(Y :: Vector{T}) where T
        set = Set{T}()
        for y in Y
            push!(set, y)
        end
        list = collect(set)
        return assign(Y, list)
    end

    @inline function zero_one(ns, n)
        return 1.0 - maximum(ns) / n
    end

    @inline function gini(ns, n)
        s = 0.0
        @simd for k in ns
            s += k * (n - k)
        end
        return s / (n * n)
    end

    # returns the entropy of ns/n
    @inline function entropy(ns, n)
        s = 0.0
        @simd for k in ns
            if k > 0
                s += k * log(k)
            end
        end
        return log(n) - s / n
    end

    # adapted from the Julia Base.Sort Library
    @inline function partition!(v, w, pivot, region)
        i, j = 1, length(region)
        r_start = region.start - 1
        @inbounds while true
            while w[i] <= pivot; i += 1; end;
            while w[j]  > pivot; j -= 1; end;
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
    function insert_sort!(v, w, lo, hi, offset)
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

    @inline function _selectpivot!(v, w, lo, hi, offset)
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
    @inline function _bi_partition!(v, w, lo, hi, offset)
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
    function q_bi_sort!(v, w, lo, hi, offset)
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


    # The code function below is a small port from numpy's library
    # library which is distributed under the 3-Clause BSD license.
    # The rest of DecisionTree.jl is released under the MIT license.

    # ported by Poom Chiarawongse <eight1911@gmail.com>

    # this is the code for efficient generation
    # of hypergeometric random variables ported from numpy.random
    function hypergeometric(good, bad, sample, rng)

        @inline function loggam(x)
            x0 = x
            n = 0
            if (x == 1.0 || x == 2.0)
                return 0.0
            elseif x <= 7.0
                n = Int(floor(7 - x))
                x0 = x + n
            end
            x2 = 1.0 / (x0*x0)
            xp = 6.2831853071795864769252867665590 # Tau
            gl0 = -1.39243221690590e+00
            gl0 = gl0 * x2 + 1.796443723688307e-01
            gl0 = gl0 * x2 - 2.955065359477124e-02
            gl0 = gl0 * x2 + 6.410256410256410e-03
            gl0 = gl0 * x2 - 1.917526917526918e-03
            gl0 = gl0 * x2 + 8.417508417508418e-04
            gl0 = gl0 * x2 - 5.952380952380952e-04
            gl0 = gl0 * x2 + 7.936507936507937e-04
            gl0 = gl0 * x2 - 2.777777777777778e-03
            gl0 = gl0 * x2 + 8.333333333333333e-02
            gl = gl0/x0 + 0.5*log(xp) + (x0-0.5)*log(x0) - x0
            if x <= 7.0
                @simd for k in 1:n
                    gl -= log(x0 - k)
                end
            end
            return gl
        end

        @inline function hypergeometric_hyp(good, bad, sample)
            d1 = bad + good - sample
            d2 = min(bad, good)

            Y = d2
            K = sample
            while Y > 0
                Y -= floor(UInt, rand(rng) + Y/(d1 + K))
                K -= 1
                if K == 0
                    break
                end
            end

            Z = d2 - Y
            return if good > bad
                sample - Z
            else
                Z
            end
        end

        @inline function hypergeometric_hrua(good, bad, sample)
            mingoodbad = min(good, bad)
            maxgoodbad = max(good, bad)
            popsize = good + bad
            m = min(sample, popsize - sample)
            d4 = mingoodbad / popsize
            d5 = 1.0 - d4
            d6 = m*d4 + 0.5
            d7 = sqrt((popsize - m) * sample * d4 * d5 / (popsize - 1) + 0.5)
            # d8 = 2*sqrt(2/e) * d7 + (3 - 2*sqrt(3/e))
            d8 = 1.7155277699214135*d7 + 0.8989161620588988
            d9 = floor(UInt, (m + 1) * (mingoodbad + 1) / (popsize + 2))
            d10 = (loggam(d9+1) + loggam(mingoodbad-d9+1) + loggam(m-d9+1) +
                   loggam(maxgoodbad-m+d9+1))
            d11 = min(m+1, mingoodbad+1, floor(UInt, d6+16*d7))

            while true
                X = rand(rng)
                Y = rand(rng)
                W = d6 + d8*(Y - 0.5)/X

                (W < 0.0 || W >= d11) && continue
                Z = floor(Int, W)
                T = d10 - (loggam(Z+1) + loggam(mingoodbad-Z+1) + loggam(m-Z+1) +
                           loggam(maxgoodbad-m+Z+1))
                (X*(4.0-X)-3.0) <= T && break
                (X*(X-T) >= 1) && continue
                (2.0*log(X) <= T) && break
            end

            if good > bad
                Z = m - Z
            end

            return if m < sample
                good - Z
            else
                Z
            end
        end

        return if sample > 10
            hypergeometric_hrua(good, bad, sample)
        else
            hypergeometric_hyp(good, bad, sample)
        end
    end

end

