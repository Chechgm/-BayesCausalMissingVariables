""" This module contains the required functions to explore the data.

Available functions:
- scatter
- density
- posterior_vs_true

sns.plotting_context("paper", rc={"font.size": 15, "axes.titlesize": 15,
                                           "axes.labelsize": 15, "legend.fontsize": 12,
                                           "lines.markersize": 8, "xtick.labelsize": 10,
                                           "ytick.labelsize": 10}):
"""
from typing import Dict, List
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def scatter(data_dictionary: Dict, var_names: List, save_directory: str):
    """ Creates a plot for the bounds, the analytic, and the maxent causal effects.
    """
    x = data_dictionary[var_names[0]]
    y = data_dictionary[var_names[1]]
    with plt.style.context('ggplot'):
        fig, ax = plt.subplots()
        ax.scatter(x, y)

        ax.set_xlabel(var_names[0])
        ax.set_ylabel(var_names[1])

        plt.savefig(save_directory, bbox_inches="tight", dpi=300)
        plt.close()


def density(data_dictionary: Dict, var_name: str, save_directory: str):
    """ Creates a plot for the bounds, the analytic, and the maxent causal effects.
    """
    x = data_dictionary[var_name]
    with plt.style.context('ggplot'):
        fig, ax = plt.subplots()
        ax.hist(x, bins=20, density=True)
        ax.set_title(f'Histogram of {var_name}');

        plt.savefig(save_directory, bbox_inches="tight", dpi=300)
        plt.close()


def posterior_vs_true(posterior: np.array, true: np.array, save_directory: str):
    """ For the first n_x latent variables, we plot n_y posterior samples and the respective true value.
    """
    n_x = 20
    n_y = 100
    jitter = np.random.uniform(0, 0.2, n_y)
    
    with plt.style.context('ggplot'):
        fig, ax = plt.subplots()
        for i in range(n_x):
            ax.scatter(i+jitter, posterior[i,:n_y], label="Posterior", color="blue", alpha=0.5)
            ax.scatter(i, true[i], label="True", color="red")

        #ax.legend()
        #plt.legend(frameon=False, loc="upper left")
        
        ax.set_title(f'Posterior and true values for u');

        plt.savefig(save_directory, bbox_inches="tight", dpi=300)
        plt.close()