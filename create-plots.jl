# create_plots.jl
# Jessica Dai, Paula Gradu, Chris Harshaw
# UC Berkeley, May 2023
#
# This file creates and saves plots and 
# tables from the simulation data.

using CSV
using DataFrames
using Distributions
using Statistics
using Random
using PyPlot 

num_samples = 50000
# num_samples = 1000
ci_val = 0.05

# get the different step size values
step_size_vals = [0.25, 0.5, 1.0, 2.0, 4.0]
orig_step_size = step_size_vals[3]
orig_ogd_col_name = "norm_ogd_var_$orig_step_size"

# plot formatting
linewidth = 9
legend_fontsize = 35
axis_label_fontsize = 38
tick_fontsize = 24
figsize = (16,9)

# create names for the plots + tables
var_plot_fn = "data/GM/created-figures/var_plot.pdf"                    # plot 1
corrupt_var_plot_fn = "data/GM/created-figures/corrupt_var_plot.pdf"    # plot 2
normality_plot_fn = "data/GM/created-figures/normality.pdf"             # plot 3
intervals_fn = "data/GM/created-figures/$(ci_val)_interval-table.csv"   # plot 4 (table)
var_plot_w_ss_comparison_fn = "data/GM/created-figures/var_plot_ss_comp.pdf"                    # plot 5 (plot 1 w/ ss comparison)
corrupt_var_plot_w_ss_comparison_fn = "data/GM/created-figures/corrupt_var_plot_ss_comp.pdf"    # plot 6 (plot 2 w/ ss comparison)
var_plot_dbcd_fn = "data/GM/created-figures/var_plot_dbcd.pdf"                                  # plot 7 (plot 1 w/ DBCD)
corrupt_var_plot_dbcd_fn = "data/GM/created-figures/corrupt_var_plot_dbcd.pdf"                  # plot 8 (plot 2 w/ DBCD)
unnorm_var_plot_DCB_fun = "data/GM/created-figures/unnormalized_var_plot_dcb.pdf"               # plot 9 (unnormalized w/ DBCD)
unnorm_var_plot_all_fun = "data/GM/created-figures/unnormalized_var_plot_all.pdf"               # plot 10 (unnormalized w/ all)

############################################
# PLOT 1: variance plot
############################################
println("Creating the first plot...")

# load the data
var_df = CSV.read("data/GM/created-data/variance$(num_samples)_samples.csv", DataFrame)

# export the relevant columns
T_vals = var_df.T_vals
norm_bern_var = var_df.norm_bern_var
norm_ney_var = var_df.norm_ney_var
norm_etc_var = var_df.norm_etc_var
norm_ogd_var = var_df[!,orig_ogd_col_name]

# plot them 
figure(1, figsize=figsize)
plot(T_vals, norm_bern_var, label="Bernoulli", color="black", linestyle="dashed", linewidth=linewidth)
plot(T_vals, norm_ney_var, label="Neyman", color="red", linestyle="dashed", linewidth=linewidth)
plot(T_vals, norm_etc_var, label="ETC", color="green", linewidth=linewidth)
plot(T_vals, norm_ogd_var, label="ClipOGD", color="blue", linewidth=linewidth)
xlabel(L"Number of Rounds ($T$)", fontsize=axis_label_fontsize)
ylabel(L"Normalized Variance $T \cdot $ Var($\widehat{\tau}$)", fontsize=axis_label_fontsize)
xticks(fontsize=tick_fontsize)
yticks(fontsize=tick_fontsize)
legend(fontsize=legend_fontsize)

savefig(var_plot_fn, format="pdf", bbox_inches="tight")

############################################
# PLOT 2: corrupted variance plot
############################################
println("Creating the second plot...")

# load the data
var_df = CSV.read("data/GM/created-data/variance_corrupt$(num_samples)_samples.csv", DataFrame)

# export the relevant columns
T_vals = var_df.T_vals
norm_bern_var = var_df.norm_bern_var
norm_ney_var = var_df.norm_ney_var
norm_etc_var = var_df.norm_etc_var
norm_ogd_var = var_df[!,orig_ogd_col_name]

# plot them 
figure(2, figsize=figsize)
plot(T_vals, norm_bern_var, label="Bernoulli", color="black", linestyle="dashed", linewidth=linewidth)
plot(T_vals, norm_ney_var, label="Neyman", color="red", linestyle="dashed", linewidth=linewidth)
plot(T_vals, norm_etc_var, label="ETC", color="green", linewidth=linewidth)
plot(T_vals, norm_ogd_var, label="ClipOGD", color="blue", linewidth=linewidth)
xlabel(L"Number of Rounds ($T$)", fontsize=axis_label_fontsize)
ylabel(L"Normalized Variance $T \cdot $ Var($\widehat{\tau}$)", fontsize=axis_label_fontsize)
xticks(fontsize=tick_fontsize)
yticks(fontsize=tick_fontsize)
legend(fontsize=legend_fontsize)

savefig(corrupt_var_plot_fn, format="pdf", bbox_inches="tight")

############################################
# PLOT 3: asymptotic normality
############################################
println("Creating the third plot...")

# load the data
sample_df = CSV.read("data/GM/created-data/estimators$(num_samples)_samples.csv", DataFrame)
po_df = CSV.read("data/GM/created-data/po-df.csv", DataFrame)

# obtain studentized estimator 
ate = mean( po_df.y1 - po_df.y0 )
emp_std = std(sample_df.ogd_eate_vals)
student_eate_vals = (ate .- sample_df.ogd_eate_vals) ./ emp_std

figure(3, figsize=figsize)
hist(student_eate_vals, bins=50)
xlabel("Studentized Adaptive HT Values", fontsize=axis_label_fontsize)
ylabel("Counts", fontsize=axis_label_fontsize)
xticks(fontsize=tick_fontsize)
yticks(fontsize=tick_fontsize)

savefig(normality_plot_fn, format="pdf", bbox_inches="tight")

############################################
# PLOT 4: coverage of intervals
############################################
println("Creating the final table...")

cheby_factor = ci_val^(-0.5)
normal_factor = quantile.(Normal(), 1.0 - ci_val/2.0 )

# create 8 new columns for width and coverage of Bernoulli + Chebyshev intervals under OGD and Bernoulli designs 
sample_df.bern_cheby_width = 2.0 * cheby_factor * sqrt.(sample_df.bern_evar_vals)
sample_df.bern_cheby_cover = abs.(ate .- sample_df.bern_eate_vals) .<= cheby_factor * sqrt.(sample_df.bern_evar_vals)
sample_df.bern_norm_width = 2.0 * normal_factor * sqrt.(sample_df.bern_evar_vals)
sample_df.bern_norm_cover = abs.(ate .- sample_df.bern_eate_vals) .<= normal_factor * sqrt.(sample_df.bern_evar_vals)

sample_df.ogd_cheby_width = 2.0 * cheby_factor * sqrt.(sample_df.ogd_evar_vals)
sample_df.ogd_cheby_cover = abs.(ate .- sample_df.ogd_eate_vals) .<= cheby_factor * sqrt.(sample_df.ogd_evar_vals)
sample_df.ogd_norm_width = 2.0 * normal_factor * sqrt.(sample_df.ogd_evar_vals)
sample_df.ogd_norm_cover = abs.(ate .- sample_df.ogd_eate_vals) .<= normal_factor * sqrt.(sample_df.ogd_evar_vals)

# average all columns over the rows
summary_df = describe(sample_df)

# save this dataframe as a csv
CSV.write(intervals_fn, summary_df)

############################################
# PLOT 5: Fig 1 w/ step size comparison
############################################
println("Creating the Plot 5 (re-creation of Plot 1)...")

# load the data
var_df = CSV.read("data/GM/created-data/variance$(num_samples)_samples.csv", DataFrame)

# export the relevant columns
T_vals = var_df.T_vals
norm_bern_var = var_df.norm_bern_var
norm_ney_var = var_df.norm_ney_var
norm_ogd_var = var_df[!,orig_ogd_col_name]

# plot them 
figure(5, figsize=figsize)
plot(T_vals, norm_bern_var, label="Bernoulli", color="black", linestyle="dashed", linewidth=linewidth)
plot(T_vals, norm_ney_var, label="Neyman", color="red", linestyle="dashed", linewidth=linewidth)
plot(T_vals, norm_ogd_var, label="ClipOGD", color="blue", linewidth=linewidth)
for (k,ss_val) in enumerate(step_size_vals)
    if ss_val ==orig_step_size
        continue
    end
    col_name = "norm_ogd_var_$ss_val"
    norm_ogd_var_ss = var_df[!,col_name]
    label_str = "Clip-OGD ($ss_val)"
    plot(T_vals, norm_ogd_var_ss, label=label_str, linewidth=linewidth)
end
xlabel(L"Number of Rounds ($T$)", fontsize=axis_label_fontsize)
ylabel(L"Normalized Variance $T \cdot $ Var($\widehat{\tau}$)", fontsize=axis_label_fontsize)
xticks(fontsize=tick_fontsize)
yticks(fontsize=tick_fontsize)
legend(fontsize=legend_fontsize)

savefig(var_plot_w_ss_comparison_fn, format="pdf", bbox_inches="tight")

############################################
# PLOT 6: Plot 2 2 w/ step size comparison
############################################
println("Creating the Plot 6 (re-creation of Plot 2)...")

# load the data
var_df = CSV.read("data/GM/created-data/variance_corrupt$(num_samples)_samples.csv", DataFrame)

# export the relevant columns
T_vals = var_df.T_vals
norm_bern_var = var_df.norm_bern_var
norm_ney_var = var_df.norm_ney_var
norm_ogd_var = var_df[!,orig_ogd_col_name]

# plot them 
figure(6, figsize=figsize)
plot(T_vals, norm_bern_var, label="Bernoulli", color="black", linestyle="dashed", linewidth=linewidth)
plot(T_vals, norm_ney_var, label="Neyman", color="red", linestyle="dashed", linewidth=linewidth)
plot(T_vals, norm_ogd_var, label="ClipOGD", color="blue", linewidth=linewidth)
for (k,ss_val) in enumerate(step_size_vals)
    if ss_val ==orig_step_size
        continue
    end
    col_name = "norm_ogd_var_$ss_val"
    norm_ogd_var_ss = var_df[!,col_name]
    label_str = "Clip-OGD ($ss_val)"
    plot(T_vals, norm_ogd_var_ss, label=label_str, linewidth=linewidth)
end
xlabel(L"Number of Rounds ($T$)", fontsize=axis_label_fontsize)
ylabel(L"Normalized Variance $T \cdot $ Var($\widehat{\tau}$)", fontsize=axis_label_fontsize)
xticks(fontsize=tick_fontsize)
yticks(fontsize=tick_fontsize)
legend(fontsize=legend_fontsize)

savefig(corrupt_var_plot_w_ss_comparison_fn, format="pdf", bbox_inches="tight")


############################################
# PLOT 7: Plot 1 w/ DBCD
############################################
println("Creating Plot 7 (Plot 1 w/ DBCD)...")

# load the data
var_df = CSV.read("data/GM/created-data/variance$(num_samples)_samples.csv", DataFrame)

# export the relevant columns
T_vals = var_df.T_vals
norm_bern_var = var_df.norm_bern_var
norm_ney_var = var_df.norm_ney_var
norm_etc_var = var_df.norm_etc_var
norm_ogd_var = var_df[!,orig_ogd_col_name]
norm_dbcd_var = var_df.norm_dbcd_var

# plot them 
figure(7, figsize=figsize)
plot(T_vals, norm_bern_var, label="Bernoulli", color="black", linestyle="dashed", linewidth=linewidth)
plot(T_vals, norm_ney_var, label="Neyman", color="red", linestyle="dashed", linewidth=linewidth)
plot(T_vals, norm_etc_var, label="ETC", color="green", linewidth=linewidth)
plot(T_vals, norm_dbcd_var, label="DBCD", color="purple", linewidth=linewidth)
plot(T_vals, norm_ogd_var, label="ClipOGD", color="blue", linewidth=linewidth)
xlabel(L"Number of Rounds ($T$)", fontsize=axis_label_fontsize)
ylabel(L"Normalized Variance $T \cdot $ Var($\widehat{\tau}$)", fontsize=axis_label_fontsize)
xticks(fontsize=tick_fontsize)
yticks(fontsize=tick_fontsize)
legend(fontsize=legend_fontsize)

savefig(var_plot_dbcd_fn, format="pdf", bbox_inches="tight")

############################################
# PLOT 8: Plot 2 w/ DBCD
############################################
println("Creating Plot 8 (Plot 2 w/ DBCD)...")

# load the data
var_df = CSV.read("data/GM/created-data/variance_corrupt$(num_samples)_samples.csv", DataFrame)

# export the relevant columns
T_vals = var_df.T_vals
norm_bern_var = var_df.norm_bern_var
norm_ney_var = var_df.norm_ney_var
norm_etc_var = var_df.norm_etc_var
norm_dbcd_var = var_df.norm_dbcd_var
norm_ogd_var = var_df[!,orig_ogd_col_name]

# plot them 
figure(8, figsize=figsize)
plot(T_vals, norm_bern_var, label="Bernoulli", color="black", linestyle="dashed", linewidth=linewidth)
plot(T_vals, norm_ney_var, label="Neyman", color="red", linestyle="dashed", linewidth=linewidth)
plot(T_vals, norm_etc_var, label="ETC", color="green", linewidth=linewidth)
plot(T_vals, norm_dbcd_var, label="DBCD", color="purple", linewidth=linewidth)
plot(T_vals, norm_ogd_var, label="ClipOGD", color="blue", linewidth=linewidth)
xlabel(L"Number of Rounds ($T$)", fontsize=axis_label_fontsize)
ylabel(L"Normalized Variance $T \cdot $ Var($\widehat{\tau}$)", fontsize=axis_label_fontsize)
xticks(fontsize=tick_fontsize)
yticks(fontsize=tick_fontsize)
legend(fontsize=legend_fontsize)

savefig(corrupt_var_plot_dbcd_fn, format="pdf", bbox_inches="tight")

############################################
# PLOT 9 + 10: Unnormalized Variance
############################################
println("Creating Plot 9 + 10 (unnormalized variance comparison)...")

# load the data
var_df = CSV.read("data/GM/created-data/variance$(num_samples)_samples.csv", DataFrame)

# export the relevant columns
T_vals = var_df.T_vals
norm_bern_var = var_df.norm_bern_var
norm_ney_var = var_df.norm_ney_var
norm_etc_var = var_df.norm_etc_var
norm_dbcd_var = var_df.norm_dbcd_var
norm_exp3_var = var_df.norm_exp3_var
norm_ogd_var = var_df[!,orig_ogd_col_name]


# plot them 
figure(9, figsize=figsize)
plot(T_vals, norm_bern_var ./ T_vals, label="Bernoulli", color="black", linestyle="dashed", linewidth=linewidth)
plot(T_vals, norm_ney_var ./ T_vals, label="Neyman", color="red", linestyle="dashed", linewidth=linewidth)
plot(T_vals, norm_etc_var ./ T_vals, label="ETC", color="green", linewidth=linewidth)
plot(T_vals, norm_dbcd_var ./ T_vals, label="DBCD", color="purple", linewidth=linewidth)
plot(T_vals, norm_ogd_var ./ T_vals, label="ClipOGD", color="blue", linewidth=linewidth)
xlabel(L"Number of Rounds ($T$)", fontsize=axis_label_fontsize)
ylabel(L"Variance Var($\widehat{\tau}$)", fontsize=axis_label_fontsize)
xticks(fontsize=tick_fontsize)
yticks(fontsize=tick_fontsize)
legend(fontsize=legend_fontsize)

savefig(unnorm_var_plot_DCB_fun, format="pdf", bbox_inches="tight")

# add one more line here
plot(T_vals, norm_exp3_var ./ T_vals, label="EXP3", color="orange", linewidth=linewidth)
savefig(unnorm_var_plot_all_fun, format="pdf", bbox_inches="tight")