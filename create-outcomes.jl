# create_outcomes.jl
# Jessica Dai, Paula Gradu, Chris Harshaw
# UC Berkeley, May 2023
#
# This file loads the dataset from Groh & MacKenzie (2016)
# cleans the data, imputes missing outcomes, and copies 
# units for larger sample size.

using CSV
using DataFrames
using Random


############################################
# IMPUTATION PARAMETERS + OUTPUT FILE PATH
# (user decides these)
###########################################

# set the parameters for generating data
ite = 90000
std_noise = 5000
num_rep = 5

# output file 
output_file = "data/GM/created-data/po-df.csv"

############################################
# IMPUTATION 
# (don't change this)
###########################################

# set the random seed for reproducability
Random.seed!(123)

# load the CSV file as a dataframe
println("Loading the original CSV file...")
df = CSV.read("data/GM/Groh2016-data.csv", DataFrame)

# remove the missing entries 
println("Cleaning missing data...")
df = df[completecases(df), :]

# Outcome 3 is: amount invested in machinery  or equipment
println("Imputing, upsampling, normalizing data...")
outcome = "o3"

# get observed outcome + treatment
obs_outcome = df[!, Symbol(outcome)]
obs_assignment = df[!, "treat"]
T = length(obs_outcome)

# construct vector of outcomes
y1 = zeros((num_rep+1)*T)
y0 = zeros((num_rep+1)*T)
for k=1:(num_rep+1)
    for t=1:T
        y = obs_outcome[t]
        z = obs_assignment[t]
        y1[(k-1)*T + t] = (z == 1) ? y : y + ite + std_noise * randn()
        y0[(k-1)*T + t] = (z == 0) ? y : y - ite + std_noise * randn()
    end
end

# normalize the variables to be in [0,1]
max_y = maximum(vcat(y1, y0))
min_y = minimum(vcat(y1, y0))
y1 = (y1 .- min_y) ./ (max_y - min_y) 
y0 = (y0 .- min_y) ./ (max_y - min_y) 
@assert all( 0.0 .<= vcat(y1, y0) .<= 1.0 ) # confirmation

# randomly permute the values 
Random.seed!(123) # re-producability of the ordering
T = length(y1) # re-defining num samples
rand_ind = randperm(T)
y1 = y1[rand_ind]
y0 = y0[rand_ind]

# make this into a dataframe 
po_df = DataFrame()
po_df.unit = collect(1:T)
po_df.y1 = y1 
po_df.y0 = y0

# save this dataframe
println("Saving data to csv file...")
CSV.write(output_file, po_df)