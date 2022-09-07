""" File that contains a simple data exploration procedure.
"""
import pickle

from utils.plot_utils import *
from utils.utils import *


def main():
    """
    """
    simulated_data_dir = "./data/simulated_data.pkl"
    posterior_dir = "./results/ground_truth_posterior.pkl"

    # Simulated data plots
    simulated_data = data_loading(simulated_data_dir)
    
    vars = ["z", "x_1", "x_2", "u", "y"]
    # Do the density plot of the continuous variables
    for i, v  in enumerate(vars):
        density(simulated_data, v, f"./results/plots/density_{v}.png")
        # Do the scatter plot of the covariates and their outcomes (maybe here we will need to add the beta in the plot)
        for w in vars[i+1:]:
            scatter(simulated_data, [v,w], f"./results/plots/scatter_{v}_{w}.png")

    # Posterior plots
    posterior_dict = data_loading(posterior_dir)
    posterior_u = posterior_dict["u"]
    true_u = simulated_data["u"]

    posterior_vs_true(posterior_u, true_u, f"./results/plots/u_posterior_vs_true.png")


if __name__ == "__main__":
    main()