# design_funs.jl
# Jessica Dai, Paula Gradu, Chris Harshaw
# UC Berkeley, May 2023
#
# This file contains functions for 
#   1. sampling from adaptive designs
#   2. constructing estimators
#   3. creating summary stats

using Random 
using Distributions
using LinearAlgebra
using Statistics
using Revise

"""
    compute_ate(y1, y0)

Compute the average treatment effect.

# Arguments
- `y1`: outcomes under treatment (array of length `T`)
- `y0`: outcomes under control (array of length `T`)

# Output 
- `ate`: the average treatment effect.
"""
function compute_ate(y1, y0)
    T = length(y1)
    return sum(y1 - y0) / T
end

"""
    compute_adaptive_HT(obs_outcomes, obs_treatments, obs_probs)

Adaptive Horvitz--Thompson estimator of average treatment effect.

# Arguments
- `obs_outcomes`: observed outcomes (array of length `T`)
- `obs_treatments`: observed treatmens (array of length `T`)
- `obs_probs`: observed sampling probabilities (array of length `T`)

# Output 
- `eate`: the adaptive HT estimate of ATE.
"""
function compute_adaptive_HT(obs_outcomes, obs_treatments, obs_probs)

    # get dimensions
    T = length(obs_outcomes)
    @assert length(obs_treatments) == T
    @assert length(obs_probs) == T

    # compute estimate for ATE
    weighting = [ (obs_treatments[t] == 1) ? (1.0 / obs_probs[t]) : -1.0 / (1 - obs_probs[t]) for t=1:T]
    eate = (1/T) * dot(obs_outcomes, weighting)
    # eay1 = (1/T) * sum( obs_outcomes[t] * (obs_treatments[t] ==1) * (1.0 / obs_probs[t]) for t=1:T)
    # eay0 = (1/T) * sum( obs_outcomes[t] * (obs_treatments[t] ==0) * (1.0 / (1.0-obs_probs[t])) for t=1:T)
    # eate = eay1 - eay0 
    return eate
end

"""
    compute_evar(obs_outcomes, obs_treatments, obs_probs)

Neyman variance estimator using observed data.

# Arguments
- `obs_outcomes`: observed outcomes (array of length `T`)
- `obs_treatments`: observed treatmens (array of length `T`)
- `obs_probs`: observed sampling probabilities (array of length `T`)

# Output 
- `evar`: the variance estimator
"""
function compute_evar(obs_outcomes, obs_treatments, obs_probs)

    # get dimensions
    T = length(obs_outcomes)
    @assert length(obs_treatments) == T
    @assert length(obs_probs) == T

    # compute estimates of second moments
    eS1_2 = (1/T) * sum( obs_outcomes[t]^2 * (obs_treatments[t] ==1) * (1.0 / obs_probs[t]) for t=1:T)
    eS0_2 = (1/T) * sum( obs_outcomes[t]^2 * (obs_treatments[t] ==0) * (1.0 / (1.0-obs_probs[t])) for t=1:T)

    evar = (4.0 / T) * sqrt( eS1_2 * eS0_2 )
    return evar
end

"""
    compute_bernoulli_evar(obs_outcomes, obs_treatments, p)

Estimate of variance under Bernoulli randomization

# Arguments
- `obs_outcomes`: observed outcomes (array of length `T`)
- `obs_treatments`: observed treatmens (array of length `T`)
- `p`: sampling probability

# Output 
- `evar`: estimate of the variance
"""
function compute_bernoulli_evar(obs_outcomes, obs_treatments, p)

    # get dimensions
    T = length(obs_outcomes)
    @assert length(obs_treatments) == T

    # compute estimates of second moments
    eS1_2 = (1/T) * sum( obs_outcomes[t]^2 * (obs_treatments[t] ==1) * (1.0 / p) for t=1:T)
    eS0_2 = (1/T) * sum( obs_outcomes[t]^2 * (obs_treatments[t] ==0) * (1.0 / (1.0-p)) for t=1:T)

    # compute estimate of the variance from estimates of the second moments
    evar = (1/T) * ( eS1_2 * (1/p - 1.0) + eS0_2 * (1.0 / (1-p) - 1.0) + 2*sqrt(eS1_2  * eS0_2) )
    return evar
end

"""
    interval_projection(x, delta)

The projection of `x` onto the interval [`delta``, 1-`delta``].
"""
function interval_projection(x, delta)
    @assert 0 <= delta <= 0.5
    return max( delta, min( x, 1-delta ) )
end

"""
    our_design_one_run(y1, y0, alpha, eta; design_seed=[nothing])

Execute one run of our design, return observed quantities

# Arguments
- `y1`: outcomes under treatment (array of length `T`)
- `y0`: outcomes under control (array of length `T`)
- `alpha`: projection interval decay parameter
- `eta`: step size

# Optional Arguments
- `design_seed`: the random seed (for reproducability)

# Output 
- `obs_outcomes`: observed outcomes (array of length `T`)
- `obs_treatments`: observed treatmens (array of length `T`)
- `obs_probs`: observed sampling probabilities (array of length `T`)
- `obs_grads`: observed gradient estimates (array of length `T`)
"""
function our_design_one_run(y1, y0, alpha, eta; design_seed=nothing)

    # set seed for reproducability
    if !isnothing(design_seed)
        Random.seed!(design_seed)
    end

    # get dimensions
    T = length(y1)
    @assert length(y0) == T

    # initialize results for observed outcomes, treatments, probabilities
    obs_outcomes = zeros(T)
    obs_treatments = zeros(T)
    obs_probs = zeros(T)
    obs_grads = zeros(T)

    # initialize sampling prob + gradient estimate
    p = 0.5
    g = 0.0

    for t=1:T 

        # set projection parameter
        delta = 0.5 * t^( - 1.0 / alpha ) 

        # update new probability
        p = interval_projection(p - eta * g, delta)

        # sample treatment assignment
        z = (rand() < p) ? 1 : 0
        
        # observe outcome
        y = (z == 1) ? y1[t] : y0[t]

        # gradient estimator 
        g = (z == 1) ? (- y^2 / p^3) :  (y^2 / (1-p)^3)

        # keep records 
        obs_outcomes[t] = y
        obs_treatments[t] = z
        obs_probs[t] = p
        obs_grads[t] = g
    end

    # return all observed values
    return obs_outcomes, obs_treatments, obs_probs, obs_grads
end

"""
    sample_our_design(y1, y0, num_samples; alpha=nothing, eta=nothing, first_seed=[nothing])

Sample adaptive HT and variance estimators from our design.

# Arguments
- `y1`: outcomes under treatment (array of length `T`)
- `y0`: outcomes under control (array of length `T`)
- `num_samples`: the number of samples to draw 

# Optional Arguments
- `step_size_val`: step size is set to `step_size_val` divided by `sqrt(T)`
- `alpha`: projection interval decay parameter -- default: sqrt( 5 logt(`T`) )
- `eta`: step size -- default: sqrt( e^alpha / T^(1 + 5.0 / alpha)  )
- `first_seed`: the first random seed -- default: no seed set

# Output 
- `eate_vals`: adaptive HT estimates (array of length `num_samples`)
- `evar_vals`: neyman variance estimates (array of length `num_samples`)
"""
function sample_our_design(y1, y0, num_samples; step_size_val=1.0, alpha=nothing, first_seed=nothing)

    # get dimensions
    T = length(y1)
    @assert length(y0) == T

    # set parameters (if not set)
    if isnothing(alpha)
        alpha = sqrt(5.0 * log(T))
        eta = step_size_val * sqrt(1.0 / T)
    else
        eta = step_size_val * sqrt( exp(alpha) / T^(1 + 5.0 / alpha) )
    end

    # initialize arrays to store estimators (ate and variance)
    eate_vals = zeros(num_samples)
    evar_vals= zeros(num_samples)

    # run the experiment multiple times
    for iter=1:num_samples

        # keep track of seeds for reproducability
        if isnothing(first_seed)
            iter_seed = nothing
        else
            iter_seed = first_seed + iter
        end

        # sample from the design
        obs_outcomes, obs_treatments, obs_probs, obs_grads = our_design_one_run(y1, y0, alpha, eta, design_seed=iter_seed)

        # compute estimators 
        eate_vals[iter] = compute_adaptive_HT(obs_outcomes, obs_treatments, obs_probs)
        evar_vals[iter] = compute_evar(obs_outcomes, obs_treatments, obs_probs)
    end

    return eate_vals, evar_vals
end

"""
    estimate_opt_prob(obs_outcomes, obs_treatments, sample_p)

Estimate the optimal sampling probability for a commit phase.

# Arguments
- `obs_outcomes`: observed outcomes (array of length `T`)
- `obs_treatments`: observed treatmens (array of length `T`)
- `sample_p`: sample probability

# Output 
- `eprob`: estimate of the optimal sampling probability
"""
function estimate_opt_prob(obs_outcomes, obs_treatments, sample_p)

    # get dimensions
    T0 = length(obs_outcomes)
    @assert length(obs_treatments) == T0

    # compute estimates of second moments
    eS1_2 = (1/T0) * sum( obs_outcomes[t]^2 * (obs_treatments[t] ==1) * (1.0 / sample_p) for t=1:T0)
    eS0_2 = (1/T0) * sum( obs_outcomes[t]^2 * (obs_treatments[t] ==0) * (1.0 / (1.0-sample_p)) for t=1:T0)

    eprob = 1.0 / (1.0 + sqrt( (1.0 + eS0_2) / (1.0 + eS1_2) ) )
    return eprob
end

"""
    ETC_one_run(y1, y0, T0; design_seed=[nothing])

Execute one run of Explore then Commit design, return observed quantities

# Arguments
- `y1`: outcomes under treatment (array of length `T`)
- `y0`: outcomes under control (array of length `T`)
- `T0`: exploration phase

# Optional Arguments
- `design_seed`: the random seed (for reproducability)

# Output 
- `obs_outcomes`: observed outcomes (array of length `T`)
- `obs_treatments`: observed treatmens (array of length `T`)
- `obs_probs`: observed sampling probabilities (array of length `T`)
"""
function ETC_one_run(y1, y0, T0; design_seed=nothing)

    # set seed for reproducability
    if !isnothing(design_seed)
        Random.seed!(design_seed)
    end

    # get dimensions
    T = length(y1)
    @assert length(y0) == T
    @assert T0 <= T

    # initialize results for observed outcomes, treatments, probabilities
    obs_outcomes = zeros(T)
    obs_treatments = zeros(T)
    obs_probs = zeros(T)

    # set explore probability
    explore_prob = 0.5

    # Phase 1: Explore (Bernoulli)
    for t=1:T0

        # sample treatment assignment
        z = (rand() < explore_prob) ? 1 : 0
        
        # observe outcome
        y = (z == 1) ? y1[t] : y0[t]

        # keep records 
        obs_outcomes[t] = y
        obs_treatments[t] = z
        obs_probs[t] = explore_prob
    end

    # estimate the optimal exploit probablity 
    eopt_prob = estimate_opt_prob(obs_outcomes[1:T0], obs_treatments[1:T0], explore_prob)

    # Phase 2: Commit
    for t=(T0+1):T 

        # sample treatment assignment
        z = (rand() < eopt_prob) ? 1 : 0
        
        # observe outcome
        y = (z == 1) ? y1[t] : y0[t]

        # keep records 
        obs_outcomes[t] = y
        obs_treatments[t] = z
        obs_probs[t] = eopt_prob
    end

    # return all observed values
    return obs_outcomes, obs_treatments, obs_probs
end

"""
    sample_ETC_design(y1, y0, num_samples; alpha=nothing, first_seed=[nothing])

Sample adaptive HT and from Explore then Commit.

# Arguments
- `y1`: outcomes under treatment (array of length `T`)
- `y0`: outcomes under control (array of length `T`)
- `num_samples`: the number of samples to draw 

# Optional Arguments
- `alpha`: exponent for `T0 = T^(alpha)` -- default `alpha = 1/3``
- `first_seed`: the first random seed -- default: no seed set

# Output 
- `eate_vals`: adaptive HT estimates (array of length `num_samples`)
"""
function sample_ETC_design(y1, y0, num_samples; alpha=nothing, first_seed=nothing)

    # get dimensions
    T = length(y1)
    @assert length(y0) == T

    # set parameters (if not set)
    if isnothing(alpha)
        alpha = 1.0 / 3.0
    end

    T0 = Int( ceil( T^(alpha) ) )

    # initialize arrays to store estimators (ate)
    eate_vals = zeros(num_samples)

    # run the experiment multiple times
    for iter=1:num_samples

        # keep track of seeds for reproducability
        if isnothing(first_seed)
            iter_seed = nothing
        else
            iter_seed = first_seed + iter
        end

        # sample from the design
        obs_outcomes, obs_treatments, obs_probs = ETC_one_run(y1, y0, T0; design_seed=iter_seed)

        # compute estimators
        eate_vals[iter] = compute_adaptive_HT(obs_outcomes, obs_treatments, obs_probs)
    end

    return eate_vals
end

"""
    bernoulli_one_run(y1, y0, p; design_seed=[nothing])

Execute one run of bernoulli design, return observed quantities

# Arguments
- `y1`: outcomes under treatment (array of length `T`)
- `y0`: outcomes under control (array of length `T`)
- `p`: assignment probability

# Optional Arguments
- `design_seed`: the random seed (for reproducability)

# Output 
- `obs_outcomes`: observed outcomes (array of length `T`)
- `obs_treatments`: observed treatmens (array of length `T`)
- `obs_probs`: observed sampling probabilities (array of length `T`)
"""
function bernoulli_one_run(y1, y0, p; design_seed=nothing)

    # set seed for reproducability
    if !isnothing(design_seed)
        Random.seed!(design_seed)
    end

    # get dimensions
    T = length(y1)
    @assert length(y0) == T

    # initialize results for observed outcomes, treatments, probabilities
    obs_outcomes = zeros(T)
    obs_treatments = zeros(T)
    obs_probs = zeros(T)

    for t=1:T 
        # sample treatment assignment
        z = (rand() < p) ? 1 : 0

        # observe outcome
        y = (z == 1) ? y1[t] : y0[t]

        # keep records 
        obs_outcomes[t] = y
        obs_treatments[t] = z
        obs_probs[t] = p
    end

    # return all observed values
    return obs_outcomes, obs_treatments, obs_probs
end

"""
    sample_bernoulli_design(y1, y0, num_samples; p=nothing, first_seed=[nothing])

Sample adaptive HT and variance estimators from our design.

# Arguments
- `y1`: outcomes under treatment (array of length `T`)
- `y0`: outcomes under control (array of length `T`)
- `num_samples`: the number of samples to draw 

# Optional Arguments
- `p`: assignment probability -- default `p = 1/2``
- `first_seed`: the first random seed -- default: no seed set

# Output 
- `eate_vals`: adaptive HT estimates (array of length `num_samples`)
- `evar_vals`: variance estimates (array of length `num_samples`)
"""
function sample_bernoulli_design(y1, y0, num_samples; p=nothing, first_seed=nothing)

    # get dimensions
    T = length(y1)
    @assert length(y0) == T

    # set parameters (if not set)
    if isnothing(p)
        p = 0.5
    end

    # initialize arrays to store estimators
    eate_vals = zeros(num_samples)
    evar_vals = zeros(num_samples)

    # run the experiment multiple times
    for iter=1:num_samples

        # keep track of seeds for reproducability
        if isnothing(first_seed)
            iter_seed = nothing
        else
            iter_seed = first_seed + iter
        end

        # sample from the design
        obs_outcomes, obs_treatments, obs_probs = bernoulli_one_run(y1, y0, p; design_seed=iter_seed)

        # compute estimators
        eate_vals[iter] = compute_adaptive_HT(obs_outcomes, obs_treatments, obs_probs)
        evar_vals[iter] = compute_bernoulli_evar(obs_outcomes, obs_treatments, p)
    end

    return eate_vals, evar_vals
end

"""
    compute_both_variances(y1, y0)

Compute variances under the p=1/2 Bernoulli design and optimal Neyman design.

# Arguments
- `y1`: outcomes under treatment (array of length `T`)
- `y0`: outcomes under control (array of length `T`)
- `p`: sample probability

# Output 
- `bern_var`: the variance under Bernoulli
- `ney_var`: the optimal neyman variance
"""
function compute_both_variances(y1, y0)

    # get dimensions
    T = length(y1)
    @assert length(y0) == T

    # compute squared second moments
    A1 = mean(y1.^2)
    A0 = mean(y0.^2)

    # compute correlation
    po_corr = mean(y1 .* y0) / sqrt(A1 * A0)

    # Neyman variance
    ney_var = (1/T) * (2.0 * (1 + po_corr) * sqrt( A1 * A0 ))

    # bernoulli probability 
    p = 0.5
    bern_var = (1/T) * ( A1 * (1 / p - 1) + A0 * (1 / (1-p) - 1) + 2 * po_corr * sqrt(A1 * A0) )

    return ney_var, bern_var
end

#########################################################
# NOT USED ANYWHERE BESIDES NOTEBOOK - CONSIDER DELETING
#########################################################

"""
    compute_interval_stats(ate, ci_val, eate_vals, evar_vals)

Compute statistics of Normal and Chebyshev based confidence intervals.

# Arguments
- `ate`: average treatment effect 
- `ci_val`: probability for CIs, e.g. set `ci_val=0.05` for 95% confidence intervals 
- `eate_vals`: adaptive HT estimates (array of length `num_samples`)
- `evar_vals`: neyman variance estimates (array of length `num_samples`)

# Output 
- `normal_width`: width of the normal-based confidence interval
- `normal_coverage`: coverage of the normal-based confidence interval
- `chebyshev_width`: width of the chebyshev-based confidence interval
- `chebyshev_coverage`: coverage of the chebyshev-based confidence interval
"""
function compute_interval_stats(ate, ci_val, eate_vals, evar_vals)

    # get dimensions
    num_samples = length(eate_vals)
    @assert length(evar_vals) == num_samples

    # get scalings 
    normal_scaling = quantile.(Normal(), 1.0 - ci_val/2.0 )
    chebyshev_scaling = ci_val^(-0.5)

    normal_width = 0.0
    normal_coverage = 0.0
    chebyshev_width = 0.0
    chebyshev_coverage = 0.0

    # compute the empirical width and coverage of intervals
    for k=1:num_samples

        # normal 
        normal_radius = normal_scaling * sqrt( evar_vals[k] ) 
        normal_width += (1/num_samples) * ( 2.0 * normal_radius)
        normal_coverage += (1/num_samples) * ( eate_vals[k] - normal_radius <= ate <= eate_vals[k] + normal_radius )


        # chebyshev
        chebyshev_radius = chebyshev_scaling * sqrt( evar_vals[k] )
        chebyshev_width += (1/num_samples) * (2.0 * chebyshev_radius )
        chebyshev_coverage += (1/num_samples) * ( eate_vals[k] - chebyshev_radius <= ate <= eate_vals[k] + chebyshev_radius )
    end

    return normal_width, normal_coverage, chebyshev_width, chebyshev_coverage
end

"""
    compute_neyman_stuff(y1, y0)

Compute Neyman variance and probability from outcomes.

# Arguments
- `y1`: outcomes under treatment (array of length `T`)
- `y0`: outcomes under control (array of length `T`)

# Output 
- `ney_prob`: the optimal neyman probability
- `ney_var`: the optimal neyman variance
"""
function compute_neyman_stuff(y1, y0)

    # get dimensions
    T = length(y1)
    @assert length(y0) == T

    # compute squared second moments
    A1 = sum(y1.^2) / T
    A0 = sum(y0.^2) / T

    # compute correlation
    po_corr = ( dot(y1, y0) / T ) / sqrt(A1 * A0)

    # optimal probability
    ney_prob = 1.0 / (1.0 + sqrt( A0 / A1 ))

    # Neyman variance
    ney_var = (1/T) * (2.0 * (1 + po_corr) * sqrt( A1 * A0 ))

    return ney_prob, ney_var
end

"""
    compute_bernoulli_var(y1, y0, p)

Compute variance of a Bernoulli design with probability p.

# Arguments
- `y1`: outcomes under treatment (array of length `T`)
- `y0`: outcomes under control (array of length `T`)
- `p`: sample probability

# Output 
- `bern_var`: the variance under Bernoulli
- `ney_var`: the optimal neyman variance
"""
function compute_bernoulli_var(y1, y0, p)

    # get dimensions
    T = length(y1)
    @assert length(y0) == T

    # compute squared second moments
    A1 = sum(y1.^2) / T
    A0 = sum(y0.^2) / T

    # compute correlation
    po_corr = ( dot(y1, y0) / T ) / sqrt(A1 * A0)

    # bernoulli probability 
    bern_var = (1/T) * ( A1 * (1 / p - 1) + A0 * (1 / (1-p) - 1) + 2 * po_corr * sqrt(A1 * A0) )

    return bern_var
end

"""
    EXP3_one_run(y1, y0, p; design_seed=[nothing])

Execute one run of EXP3, return observed quantities

# Arguments
- `y1`: outcomes under treatment (array of length `T`)
- `y0`: outcomes under control (array of length `T`)
- `eta`: step size for EXP3

# Optional Arguments
- `design_seed`: the random seed (for reproducability)

# Output 
- `obs_outcomes`: observed outcomes (array of length `T`)
- `obs_treatments`: observed treatmens (array of length `T`)
- `obs_probs`: observed sampling probabilities (array of length `T`)
"""
function EXP3_one_run(y1, y0, eta; design_seed=nothing)

    # set seed for reproducability
    if !isnothing(design_seed)
        Random.seed!(design_seed)
    end

    # get dimensions
    T = length(y1)
    @assert length(y0) == T

    # initialize results for observed outcomes, treatments, probabilities
    obs_outcomes = zeros(T)
    obs_treatments = zeros(T)
    obs_probs = zeros(T)

    # initialize rewards
    sum_vals = [0.0, 0.0]

    for t=1:T 

        # # compute sampling probabilities from observed rewards
        # g = exp.( eta * sum_vals )
        # prob_vals = g / sum(g)
        # p = prob_vals[1]
        p = 1.0 / ( 1 + exp( eta * (sum_vals[2] - sum_vals[1])) ) # more numerically stable?

        # sample treatment assignment
        z = (rand() < p) ? 1 : 0

        # observe outcome
        y = (z == 1) ? y1[t] : y0[t]

        # update the rewards 
        sum_vals .+= 1.0
        if z == 1
            sum_vals[1] -= (1 - y) / p
        else
            sum_vals[2] -= (1 - y) / (1-p)
        end

        # keep records 
        obs_outcomes[t] = y
        obs_treatments[t] = z
        obs_probs[t] = p
    end

    # return all observed values
    return obs_outcomes, obs_treatments, obs_probs
end

"""
    sample_EXP3_design(y1, y0, num_samples; first_seed=[nothing])

Sample adaptive HT from EXP3.

# Arguments
- `y1`: outcomes under treatment (array of length `T`)
- `y0`: outcomes under control (array of length `T`)
- `num_samples`: the number of samples to draw 

# Optional Arguments
- `first_seed`: the first random seed -- default: no seed set

# Output 
- `eate_vals`: adaptive HT estimates (array of length `num_samples`)
"""
function sample_EXP3_design(y1, y0, num_samples; first_seed=nothing)

    # get dimensions
    T = length(y1)
    @assert length(y0) == T

    # set the step size according to Theorem 11.1 of Lattimore + Szepesv ́ari
    eta = sqrt( log(2) / (2*T) )

    # initialize arrays to store estimators (ate)
    eate_vals = zeros(num_samples)

    # run the experiment multiple times
    for iter=1:num_samples

        # keep track of seeds for reproducability
        if isnothing(first_seed)
            iter_seed = nothing
        else
            iter_seed = first_seed + iter
        end

        # sample from the design
        obs_outcomes, obs_treatments, obs_probs = EXP3_one_run(y1, y0, eta; design_seed=iter_seed)

        # compute estimators
        eate_vals[iter] = compute_adaptive_HT(obs_outcomes, obs_treatments, obs_probs)
    end

    return eate_vals
end

"""
    update_online_estimates(existing_aggregate, new_value)

Update online estimates for mean and standard deviation.

For description of the algorithm, see 
https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm

# Arguments
- `existing_aggregate`: array containing
  - `existing_est[1]` is the count
  - `existing est[2]` is the mean 
  - `existing_est[3]` is the M2 value

# Output 
- none, modifies in place
"""

function update_online_estimates!(existing_aggregate, new_value)

    # get the values
    count, mean, M2 = existing_aggregate

    # update count
    count += 1

    # update mean + M2
    delta = new_value - mean
    mean += delta / count
    delta2 = new_value - mean
    M2 += delta * delta2

    # put them back in the array
    existing_aggregate[:] = [count, mean, M2]
end

"""
    DBCD_one_run(y1, y0, p; design_seed=[nothing])

Execute one run of DBCD (Doubly Biased Coin Design), return observed quantities.

For description of the algorithm, see Section 6 in Eisele (1994).

# Arguments
- `y1`: outcomes under treatment (array of length `T`)
- `y0`: outcomes under control (array of length `T`)
- `burn_num`: number of initial outcomes we should observe from each arm

# Optional Arguments
- `design_seed`: the random seed (for reproducability)

# Output 
- `obs_outcomes`: observed outcomes (array of length `T`)
- `obs_treatments`: observed treatmens (array of length `T`)
- `obs_probs`: observed sampling probabilities (array of length `T`)
"""
function DBCD_one_run(y1, y0, burn_num; design_seed=nothing)

    # set seed for reproducability
    if !isnothing(design_seed)
        Random.seed!(design_seed)
    end

    # get dimensions
    T = length(y1)
    @assert length(y0) == T

    # initialize results for observed outcomes, treatments, probabilities
    obs_outcomes = zeros(T)
    obs_treatments = zeros(T)
    obs_probs = zeros(T)

    # burn in period
    z_burn = vcat( zeros(burn_num), ones(burn_num) )
    shuffle!(z_burn)
    for t=1:2*burn_num

        # get assignment + outcome
        z = z_burn[t]
        y = (z == 1) ? y1[t] : y0[t]

        # keep records 
        obs_outcomes[t] = y
        obs_treatments[t] = z
        obs_probs[t] = 0.5
    end

    # initialize the estimates of mean and standard deviation 
    obs_out_1 = [obs_outcomes[s] for s=1:2*burn_num if obs_treatments[s] == 1]
    obs_out_0 = [obs_outcomes[s] for s=1:2*burn_num if obs_treatments[s] == 0]
    mean_1 = mean(obs_out_1)
    mean_0 = mean(obs_out_0)
    M2_1 = sum((obs_out_1 .- mean_1).^2)
    M2_0 = sum((obs_out_0 .- mean_0).^2)
    existing_aggregate_1 = [burn_num, mean_1, M2_1]
    existing_aggregate_0 = [burn_num, mean_0, M2_0]

    # for all remaining iterations
    for t=(2*burn_num + 1):T 

        # get standard deviation estimates + counts 
        m1 = existing_aggregate_1[1]
        std_1 = sqrt( existing_aggregate_1[3] / existing_aggregate_1[1])
        std_0 = sqrt( existing_aggregate_0[3] / existing_aggregate_0[1])
            
        # pick q + the probability
        q = max( 1 - (std_0 / std_1) * (m1 / t)  , 0)
        p = min(max(0.0, q) , 1.0)

        # sample treatment assignment
        z = (rand() < p) ? 1 : 0
       
        # observe outcome
        y = (z == 1) ? y1[t] : y0[t]

        # update online estimators 
        if z == 1
            update_online_estimates!(existing_aggregate_1, y)
        else
            update_online_estimates!(existing_aggregate_0, y)
        end

        # keep records 
        obs_outcomes[t] = y
        obs_treatments[t] = z
        obs_probs[t] = p
    end

    # return all observed values
    return obs_outcomes, obs_treatments, obs_probs
end

"""
    sample_DBCD_design(y1, y0, num_samples; first_seed=[nothing])

Sample adaptive HT from DBCD (Doubly Biased Coin Design).

# Arguments
- `y1`: outcomes under treatment (array of length `T`)
- `y0`: outcomes under control (array of length `T`)
- `num_samples`: the number of samples to draw 

# Optional Arguments
- `first_seed`: the first random seed -- default: no seed set

# Output 
- `eate_vals`: adaptive HT estimates (array of length `num_samples`)
"""
function sample_DBCD_design(y1, y0, num_samples; first_seed=nothing)

    # get dimensions
    T = length(y1)
    @assert length(y0) == T

    # set the burn-in length to be 2
    burn_num = 2

    # initialize arrays to store estimators (ate)
    eate_vals = zeros(num_samples)

    # run the experiment multiple times
    for iter=1:num_samples

        # keep track of seeds for reproducability
        if isnothing(first_seed)
            iter_seed = nothing
        else
            iter_seed = first_seed + iter
        end

        # sample from the design
        obs_outcomes, obs_treatments, obs_probs = DBCD_one_run(y1, y0, burn_num; design_seed=iter_seed)

        # compute estimators
        eate_vals[iter] = compute_adaptive_HT(obs_outcomes, obs_treatments, obs_probs)
    end

    return eate_vals
end