""" This module contains the pipeline to fit the models to the data.
"""
import numpy as np

import arviz as az
import stan

from utils.utils import *


def main():
    """ 
    TODO: consider abstracting more the other elements of this function
    TODO: consider using argparse for this main
    TODO: Consider taking the model as argument to this function and making a directory for each model
    """
    model = "parametric_ground_truth"
    model_dir = f"./STAN/{model}.stan"
    data_dir = "./data/simulated_data.pkl"
    results_dir = f"./results/{model}_posterior.pkl"

    N = 100 # How many observations to use?

    simulated_data = data_loading(data_dir)
    model_data = {
        "N": N,
        "y": simulated_data["y"][:N],
        "x_1": simulated_data["x_1"][:N],
        "x_2": simulated_data["x_2"][:N],
        "z": simulated_data["z"][:N],
        "mu_b_0_u": simulated_data["beta_0_u"],
        "mu_b_1_u": simulated_data["beta_1_u"],
        "mu_b_2_u": simulated_data["beta_2_u"],
        "mu_b_0_y": simulated_data["beta_0_y"],
        "mu_b_1_y": simulated_data["beta_1_y"],
        "mu_b_2_y": simulated_data["beta_2_y"],
        "mu_b_u_y": simulated_data["beta_u_y"],
        "mu_b_z_y": simulated_data["beta_z_y"],
        "sd_priors": np.array([.5])
    }
    model_definition = load_model(model_dir)
    
    posterior = stan.build(model_definition, data=model_data, random_seed=2)
    fit = posterior.sample(num_chains=4, num_samples=500)

    data_dictionary = {
        "u": fit["u"],
        "beta_0_y": fit["beta_0_y"],
        "beta_1_y": fit["beta_1_y"],
        "beta_2_y": fit["beta_2_y"],
        "beta_u_y": fit["beta_u_y"],
        "beta_z_y": fit["beta_z_y"],
        "sigma_y": fit["sigma_y"],
        "beta_0_u": fit["beta_0_u"],
        "beta_1_u": fit["beta_1_u"],
        "beta_2_u": fit["beta_2_u"],
        "sigma_u": fit["sigma_u"],
    }
    
    data_saving(data_dictionary, results_dir)

    # Saving the posterior table TODO: consider packaging this into a function.
    parameter_summary = az.summary(fit).head(10)
    parameter_summary["true"] = 0
    for i in parameter_summary.index:
        parameter_summary.loc[i, "true"] = np.round(simulated_data[i], 3)
    
    with open(f"./results/table_{model}.txt", "w") as f:
        f.write(parameter_summary.to_latex())


if __name__ == "__main__":
    main()
