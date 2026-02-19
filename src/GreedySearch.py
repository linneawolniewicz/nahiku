import time
import torch
import gpytorch
import warnings
import numpy as np
import matplotlib.pyplot as plt

from nahiku.src.Search import Search

from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.ndimage import minimum_filter1d

class GreedySearch(Search):
    """
        This class implements a greedy search algorithm to identify anomalous intervals in a time series using Gaussian Processes.

        Method:
            1. Perform GP regression on the time series.
            2. Find the most significant outlier interval (based on sum of residuals) of length len_deviant.
            3. Exclude the outlier interval and redo regression. See if GP improves by some threshold.
            4. Expand outlier interval in both directions by expansion_param and redo step 3.
            5. Repeat step 4 as long as GP improves the fit by some threshold.
            6. If no improvement, define anomaly signal as the difference between data and regression in the outlier interval of points.
            7. Repeat steps 2-6 while there are still points above the num_sigma_threshold.
    """
    def __init__(
        self,
        x,                 # Map to Search.x
        y,                 # Map to Search.y
        dominant_period,   # Map to Search.dominant_period
        device="cpu",      # Map to Search.device
        which_grow_metric="mll",
        y_err=None,
        num_sigma_threshold=3,
        expansion_param=1,
        len_deviant=1,
    ):
        """
        Initialize the GreedySearch class and the base Search class with the provided parameters.
        
        :param x (np.ndarray): x array of the light curve
        :param y (np.ndarray): y array of the light curve
        :param dominant_period (float): dominant period of the light curve
        :param device (str): device to use for GP modeling (default: "cpu")
        :param which_grow_metric (str): Metric to use for evaluating improvement when expanding the anomalous region. Options are 'nlpd', 'msll', 'rmse', 'mll'. Default is 'mll'.
        :param y_err (np.ndarray or None): 1D array of observational errors. If None, assumes zero error for all points.
        :param num_sigma_threshold (float): Threshold in terms of standard deviations for identifying anomalies. Default is 3.
        :param expansion_param (int): Number of indices to expand the anomalous region on each side during the greedy search. Default is 1.
        :param len_deviant (int): Length of the interval to consider for identifying the most significant outlier. Default is 1.
        """

        # Initialize the Base Search class
        # This handles self.x, self.y, self.x_tensor, self.y_tensor, and self.device
        super().__init__(
            x=x,
            y=y,
            dominant_period=dominant_period,
            device=device
        )

        # If y_err is not provided, set y_err = 0 for all points
        if y_err is None:
            print(
                "No y_err provided. Using y_err = 0 for all points. Note that y_err is only used for calculating residuals, not in GP training."
            )
            self.y_err = np.zeros_like(y)
        else:
            self.y_err = y_err

        # Save copies of x, y, and y_err
        self.x_orig = x
        self.y_orig = y
        self.y_err_orig = y_err

        # Initialize variables
        self.num_sigma_threshold = num_sigma_threshold
        self.expansion_param = expansion_param
        self.which_grow_metric = which_grow_metric

        # Check that len_deviant is valid
        if len_deviant <= 0:
            warnings.warn("len_deviant must be greater than 0. Setting len_deviant = 1.")
            self.len_deviant = 1
        else:
            self.len_deviant = len_deviant


    def search_for_anomaly(
        self,
        training_iterations=1_000,
        lr=0.01,
        which_train_metric="mll",  # 'mse' or 'mll'
        which_opt="adam",  # Which optimizer to use. Options are 'adam' or 'sgd'
        early_stopping=True,
        min_iterations=150,  # If early_stop is True, minimum number of iterations to run before stopping
        patience=1,  # If early_stop is True, number of iterations to wait before stopping
        set_noise_equal_to_var_residuals=True,
        refit=True,  # Whether to refit the GP at each iteration
        neg_anomaly_only=False, # Whether to only flag negative anomalies (i.e., dips)
        plot=False,
        anomaly_locs=None,
        detection_range=None,
        update_threshold=False,
    ):
    
        pass