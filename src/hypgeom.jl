
# The code in this file is a small port from numpy's library
# library which is distributed under the 3-Clause BSD license. 
# The rest of DecisionTree.jl is released under the MIT license. 

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
                Y -= floor(UInt64, rand(rng) + Y/(d1 + K))
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
            d9 = floor(UInt64, (m + 1) * (mingoodbad + 1) / (popsize + 2))
            d10 = (loggam(d9+1) + loggam(mingoodbad-d9+1) + loggam(m-d9+1) +
                   loggam(maxgoodbad-m+d9+1))
            d11 = min(m+1, mingoodbad+1, floor(UInt64, d6+16*d7))

            while true
                X = rand(rng)
                Y = rand(rng)
                W = d6 + d8*(Y - 0.5)/X

                (W < 0.0 || W >= d11) && continue
                Z = floor(Int64, W)
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
