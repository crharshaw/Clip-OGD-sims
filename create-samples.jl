# create_samples.jl
# Jessica Dai, Paula Gradu, Chris Harshaw
# UC Berkeley, May 2023
#
# This file samples from different designs
# and stores the relevant result in CSV files.

using CSV
using DataFrames
using Random
using Statistics

include("design-funs.jl") 

num_samples = 50000
# num_samples = 1000
num_T = 35
corrupt_num = 100
step_size_vals = [0.25, 0.5, 1.0, 2.0, 4.0]

samples_file = "data/GM/created-data/estimators$(num_samples)_samples.csv"
var_file = "data/GM/created-data/variance$(num_samples)_samples.csv"
var_corrupt_file = "data/GM/created-data/variance_corrupt$(num_samples)_samples.csv"
num_step_size = length(step_size_vals)

############################################
# LOAD OUTCOME DATA
############################################

# load potential outcomes from CSV file 
println("Loading the outcomes data frame...")
df_po = CSV.read("data/GM/created-data/po-df.csv", DataFrame)
y1 = df_po.y1
y0 = df_po.y0

# decide the values of T to run on
T = length(y1)
T_vals = collect(range(500, stop=T, length=num_T))
T_vals = Integer.(round.(T_vals))

############################################
# SIMULATIONS 1: NON-ADVERSARIAL
############################################
println("First simulation...")

# Initialize data arrays
bern_var = zeros(num_T)
ney_var = zeros(num_T)
etc_var = zeros(num_T)
exp3_var = zeros(num_T)
dbcd_var = zeros(num_T)
ogd_var = zeros(num_T, num_step_size)

# for all values of T, get the 
for (k,t) in enumerate(T_vals)

    println("\tWorking on T=$t ($k of $num_T)")

    # get outcomes up to time T
    y1_T = y1[1:t]
    y0_T = y0[1:t]

    # compute the Bernoulli + Neyman variance
    ney_var[k], bern_var[k] = compute_both_variances(y1_T, y0_T)

    # sample from ETC
    etc_eate_vals = sample_ETC_design(y1_T, y0_T, num_samples, first_seed=1)
    etc_var[k] = var(etc_eate_vals)

    # sample from EXP3
    exp3_eate_vals = sample_EXP3_design(y1_T, y0_T, num_samples, first_seed=1)
    exp3_var[k] = var(exp3_eate_vals)

    # sample from DBCD
    dbcd_eate_vals = sample_DBCD_design(y1_T, y0_T, num_samples, first_seed=1)
    dbcd_var[k] = var(dbcd_eate_vals)

    # sample from Clip-OGD with different step sizes
    for (r, ss_val) in enumerate(step_size_vals)
        ogd_eate_vals, ogd_evar_vals = sample_our_design(y1_T, y0_T, num_samples, step_size_val=ss_val, first_seed=1)
        ogd_var[k,r] = var(ogd_eate_vals)
    end

    # this is for CONFIDENCE INTERVALS (no need to change anything here)
    if k == num_T
        # sample from Bernoulli
        bern_eate_vals, bern_evar_vals = sample_bernoulli_design(y1_T, y0_T, num_samples, first_seed=1)

        # create a dataframe for all these values 
        bern_df = DataFrame()
        bern_df.seed = collect(1:num_samples)
        bern_df.bern_eate_vals = bern_eate_vals
        bern_df.bern_evar_vals = bern_evar_vals

        # create a dataframe for Clip-OGD values 
        ogd_eate_vals, ogd_evar_vals = sample_our_design(y1_T, y0_T, num_samples, step_size_val=1.0, first_seed=1)
        ogd_df = DataFrame()
        ogd_df.seed = collect(1:num_samples)
        ogd_df.ogd_eate_vals = ogd_eate_vals
        ogd_df.ogd_evar_vals = ogd_evar_vals

        # combine the two data frames 
        df_samples = innerjoin(bern_df, ogd_df, on = :seed)
        println("Saving the samples for T=$t...")
        CSV.write(samples_file, df_samples)
    end # end saving samples
end # end iterating over T_vals

# create a dataframe for variances
var_df = DataFrame()
var_df.T_vals = T_vals

var_df.norm_bern_var = T_vals .* bern_var 
var_df.norm_ney_var = T_vals .* ney_var
var_df.norm_etc_var = T_vals .* etc_var
var_df.norm_exp3_var = T_vals .* exp3_var
var_df.norm_dbcd_var = T_vals .* dbcd_var
for (r,ss_val) in enumerate(step_size_vals)
    col_name = "norm_ogd_var_$ss_val"
    var_df[!, col_name] = T_vals .* ogd_var[:,r]
end

println("Saving the variances...")
CSV.write(var_file, var_df)

############################################
# SIMULATIONS 2: ADVERSARIAL
############################################
println("\n\nSecond simulation (corrupting first 100 outcomes)...")

# create corrupted outcomes by swapping y(1) and y(0) for first few units
y1_corr = deepcopy(y1)
y0_corr = deepcopy(y0)
swap_range = 1:corrupt_num
tmp = y1_corr[swap_range]
y1_corr[swap_range] = y0_corr[swap_range]
y0_corr[swap_range] = tmp

# Initialize data arrays
bern_var = zeros(num_T)
ney_var = zeros(num_T)
etc_var = zeros(num_T)
exp3_var = zeros(num_T)
dbcd_var = zeros(num_T)
ogd_var = zeros(num_T, num_step_size)

# for all values of T, get the 
for (k,t) in enumerate(T_vals)

    println("\tWorking on T=$t ($k of $num_T)")

    # get (corrupted) outcomes up to time T
    y1_T = y1_corr[1:t]
    y0_T = y0_corr[1:t]

    # compute the Bernoulli + Neyman variance
    ney_var[k], bern_var[k] = compute_both_variances(y1_T, y0_T)

    # sample from ETC
    etc_eate_vals = sample_ETC_design(y1_T, y0_T, num_samples, first_seed=1)
    etc_var[k] = var(etc_eate_vals)

    # sample from EXP3
    exp3_eate_vals = sample_EXP3_design(y1_T, y0_T, num_samples, first_seed=1)
    exp3_var[k] = var(exp3_eate_vals)

    # sample from DBCD
    dbcd_eate_vals = sample_DBCD_design(y1_T, y0_T, num_samples, first_seed=1)
    dbcd_var[k] = var(dbcd_eate_vals)

    # sample from Clip-OGD with different step sizes
    for (r, ss_val) in enumerate(step_size_vals)
        ogd_eate_vals, ogd_evar_vals = sample_our_design(y1_T, y0_T, num_samples, step_size_val=ss_val, first_seed=1)
        ogd_var[k,r] = var(ogd_eate_vals)
    end
end

# create a dataframe for variances
corr_var_df = DataFrame()
corr_var_df.T_vals = T_vals
corr_var_df.norm_bern_var = T_vals .* bern_var 
corr_var_df.norm_ney_var = T_vals .* ney_var
corr_var_df.norm_etc_var = T_vals .* etc_var
corr_var_df.norm_exp3_var = T_vals .* exp3_var
corr_var_df.norm_dbcd_var = T_vals .* dbcd_var
for (r,ss_val) in enumerate(step_size_vals)
    col_name = "norm_ogd_var_$ss_val"
    corr_var_df[!, col_name] = T_vals .* ogd_var[:,r]
end

println("Saving the variances on corrupted inputs...")
CSV.write(var_corrupt_file, corr_var_df)