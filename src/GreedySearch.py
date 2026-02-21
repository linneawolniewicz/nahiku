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
        # It also intializes self.num_detected_anomalies, self.flagged_anomalous, self.flagged_anomalous_signal, and self.runtime
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

    def plot(self, pred_mean, left_edge, right_edge, residuals):
        """
        Plot the light curve, GP fit, and detected anomalies at each iteration of the greedy search.

        :param pred_mean (np.ndarray): Array of the GP mean predictions corresponding to self.x at the current iteration
        :param left_edge (int): Left edge index of the currently flagged anomalous region
        :param right_edge (int): Right edge index of the currently flagged anomalous region
        :param residuals (np.ndarray): Array of residuals (absolute value of observed - predicted) for the current iteration
        """

        fig, axs = plt.subplots(1, 2, sharex=True, figsize=(15, 5))

        # Plot the GP mean prediction vs. data
        axs[0].plot(
            self.x, pred_mean, lw=2, alpha=0.9, label="GP Mean Prediction"
        )
        axs[0].plot(
            self.x_orig, 
            self.y_orig, 
            ".k",
            markersize=3,
            label="Observed"
        )
        axs[0].plot(
            self.x_orig[left_edge:right_edge],
            self.y_orig[left_edge:right_edge],
            ".r", 
            markersize=5,
        )
        axs[0].plot(
            self.x_orig[(self.flagged_anomalous == 1)],
            self.y_orig[(self.flagged_anomalous == 1)],
            ".r", 
            markersize=5,
            label="Flagged as Anomalous",
        )
        axs[0].set_ylim(
            np.min(self.y_orig), np.max(self.y_orig)
        )
        axs[0].legend()
        axs[0].set_xlim(np.min(self.x_orig), np.max(self.x_orig))
        axs[0].set_xlabel("Time [days]")
        axs[0].set_ylabel("Standardized Flux")

        # Plot the residuals
        axs[1].plot(
            self.x,
            self.threshold,
            "--",
            lw=3,
            alpha=0.9,
            color="C0",
            label=f"Threshold = {self.num_sigma_threshold} "
            + r"$\sqrt{\text{var} + \text{err}^2}$",
        )
        axs[1].plot(
            self.x, 
            residuals, 
            ".k",
            markersize=3,
            label="|observed - predicted|"
        )
        axs[1].plot(
            self.x[left_edge:right_edge],
            residuals[left_edge:right_edge],
            ".r", 
            markersize=5,
            label="Flagged as Anomalous",
        )


        # Plot the max_sum_idx of the new anomaly
        axs[0].axvspan(
            self.x[left_edge],
            self.x[right_edge],
            color="green",
            alpha=0.6,
            label="New flagged anomaly",
        )
        axs[1].axvspan(
            self.x[left_edge],
            self.x[right_edge],
            color="green",
            alpha=0.6,
            label="New flagged anomaly",
        )
        axs[1].legend()
        axs[1].set_xlim(np.min(self.x_orig), np.max(self.x_orig))
        axs[1].set_xlabel("Time [days]")
        axs[1].set_ylabel("Standardized Flux")
        plt.show()


    def search_for_anomaly(
        self,
        refit=True,  
        neg_anomaly_only=False,
        pos_anomaly_only=False, 
        plot=False,
        detection_range=None,
        update_threshold=False,
        training_iterations=1_000,
        lr=0.01,
        which_train_metric="mll",  
        which_opt="adam", 
        early_stopping=True,
        min_iterations=150,  
        patience=1, 
        set_noise_equal_to_var_residuals=True,
    ):
        """
        Main function to perform the greedy search for anomalies in the time series data.
        
        :param refit (bool): Whether to refit the GP model at each iteration of the greedy search (default: True)
        :param neg_anomaly_only (bool): Whether to only flag negative anomalies (i.e., dips) instead of both positive and negative anomalies (default: False)
        :param pos_anomaly_only (bool): Whether to only flag positive anomalies (i.e., flares) instead of both positive and negative anomalies (default: False)
        :param plot (bool): Whether to the light curve, GP fit, and detected anomalies at each iteration of the greedy search (default: False)
        :param detection_range (tuple or None): Tuple specifying the range of x values to consider for anomaly detection. If None, considers the entire range of x. Default is None.
        :param update_threshold (bool): Whether to update the num_sigma_threshold after each detected anomaly
        :param training_iterations (int): maximum number of training iterations (default: 1000)
        :param lr (float): learning rate for the optimizer (default: 0.01)
        :param which_train_metric (str): Metric to use for evaluating improvement during training. Options are 'mll' for marginal log likelihood and 'mse' for mean squared error. Default is 'mll'.
        :param which_opt (str): Optimizer to use for training. Options are 'adam' and 'sgd'. Default is 'adam'.
        :param early_stopping (bool): Whether to use early stopping based on the training loss (default: True)
        :param min_iterations (int or None): Minimum number of iterations to train before considering early stopping (default: None, which sets it to training_iterations // 10)
        :param patience (int): Number of consecutive iterations with increasing loss to wait before stopping when early_stopping is True (default: 1)
        :param set_noise_equal_to_var_residuals (bool): Whether to set the likelihood noise variance equal to the variance of the residuals after training (default: False)
        """
        start_time = time.time()

        # Build GP model on full x and y data
        model, likelihood, _, _ = self.build_gp_model(x=self.x_tensor, y=self.y_tensor)

        # Train model to get initial fit
        model, likelihood, _ = self.train_gp(
            gp_model=model,
            likelihood=likelihood,
            x=self.x_tensor,
            y=self.y_tensor,
            device=self.device,
            training_iterations=training_iterations,
            lr=lr,
            which_metric=which_train_metric,
            which_opt=which_opt,
            early_stopping=early_stopping,
            min_iterations=min_iterations,
            patience=patience,
            plot=plot,
            set_noise_equal_to_var_residuals=set_noise_equal_to_var_residuals,
        )

        # Update kernel and mean with learned parameters
        kernel = model.covar_module
        mean = model.mean_module

        # Get mean prediction from the learned model
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = model(self.x_tensor)

        # Calculate the threshold for flagging anomalies based on the residuals and y_err
        residuals = np.abs(self.y - observed_pred.mean.cpu().numpy())
        residual_var = np.var(residuals)
        sum_variances = self.y_err**2 + residual_var
        self.threshold = self.num_sigma_threshold * np.sqrt(sum_variances)
        exist_points_above_threshold = True

        # Step 7 (repeat steps 2-6 while there are still points above the num_sigma_threshold)
        while exist_points_above_threshold:
            model, likelihood, _, _ = self.build_gp_model(x=self.x_tensor, y=self.y_tensor, kernel=kernel, mean=mean, likelihood=likelihood, device=self.device) # Note: initializing from previous hyperparameters

            # Re-fit the GP on non-anomalous data
            if refit:
                model, likelihood, _ = self.train_gp(
                    gp_model=model,
                    likelihood=likelihood,
                    x=self.x_tensor,
                    y=self.y_tensor,
                    device=self.device,
                    training_iterations=training_iterations,
                    lr=lr,
                    which_metric=which_train_metric,
                    which_opt=which_opt,
                    early_stopping=early_stopping,
                    min_iterations=min_iterations,
                    patience=patience,
                    plot=plot,
                    set_noise_equal_to_var_residuals=set_noise_equal_to_var_residuals,
                )

            # Get mean prediction from the learned model over x and x_orig
            model.eval()
            likelihood.eval()

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = likelihood(model(self.x_tensor))
                pred_mean = observed_pred.mean.cpu().numpy()

                pred_full_x = model(
                    torch.tensor(self.x_orig, dtype=torch.float32).to(self.device)
                ).mean.cpu().numpy()

            # Compute the minimum value in each window of size len_deviant in the residuals array
            residuals = np.abs(pred_mean - self.y)
            min_values = minimum_filter1d(residuals, size=self.len_deviant, mode="nearest")
            max_min_idx = np.argmax(min_values)

            # Intialize variables for expanding anomalous region
            left_edge = max_min_idx
            right_edge = max_min_idx + self.len_deviant
            diff_metric = float('inf')
            metric = float('inf')

            # Plot
            if plot: self.plot(pred_mean, left_edge, right_edge, residuals)

            # While the metric is decreasing, expand the anomalous edges
            while diff_metric > 0:
                # Subset x, y, and y_err by left_edge and right_edge
                # Do not need to worry about masking by anomalous, because anomalous points are removed at the end of the loop
                subset = (np.arange(len(self.x)) > right_edge) | (np.arange(len(self.x)) < left_edge)
                x_sub = torch.tensor(self.x[subset], dtype=torch.float32).to(
                    self.device
                )
                y_sub = torch.tensor(self.y[subset], dtype=torch.float32).to(
                    self.device
                )

                model, likelihood, _, _ = self.build_gp_model(x=x_sub, y=y_sub, kernel=kernel, mean=mean, likelihood=likelihood, device=self.device) # Note: initializing from previous hyperparameters

                # Re-fit the GP on non-anomalous data
                if refit:
                    model, likelihood, _ = self.train_gp(
                        gp_model=model,
                        likelihood=likelihood,
                        x=x_sub,
                        y=y_sub,
                        device=self.device,
                        training_iterations=training_iterations,
                        lr=lr,
                        which_metric=which_train_metric,
                        which_opt=which_opt,
                        early_stopping=early_stopping,
                        min_iterations=min_iterations,
                        patience=patience,
                        plot=plot,
                        set_noise_equal_to_var_residuals=set_noise_equal_to_var_residuals,
                    )

                # Predict on the subset and on the full x_orig
                model.eval()
                likelihood.eval()

                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    observed_pred_sub = likelihood(model(x_sub))
                    pred_mean_sub = observed_pred_sub.mean.cpu().numpy()
                    pred_full_x = model(
                        torch.tensor(self.x_orig, dtype=torch.float32).to(self.device)
                    ).mean.cpu().numpy()

                # Calculate metric difference
                old_metric = metric

                # NLPD loss
                if self.which_grow_metric == "nlpd":
                    metric = gpytorch.metrics.negative_log_predictive_density(
                        observed_pred_sub, y_sub
                    )

                # MSLL loss
                elif self.which_grow_metric == "msll":
                    metric = gpytorch.metrics.mean_standardized_log_loss(
                        observed_pred_sub, y_sub
                    )

                # RMSE loss
                elif self.which_grow_metric == "rmse":
                    metric = np.sqrt(
                        np.mean((pred_mean_sub - y_sub.cpu().numpy()) ** 2)
                    )

                # MLL loss
                else:
                    with torch.no_grad(), gpytorch.settings.fast_pred_var():
                        output = model(x_sub)
                        mll_func = ExactMarginalLogLikelihood(likelihood, model)
                        metric = mll_func(output, y_sub)

                diff_metric = old_metric - metric  # smaller is better

                # Expand left_edge and right_edge by expansion_param
                if left_edge >= (0 + self.expansion_param):
                    left_edge -= self.expansion_param

                if right_edge < (len(self.x) - self.expansion_param):
                    right_edge += self.expansion_param

                # Plot
                if plot: self.plot(pred_mean_sub, left_edge, right_edge, residuals)

            # Remove left_edge:right_edge from x, y, and y_err for the next iteration of the greedy search
            # Handle case where left_edge = 0 or right_edge = len(self.x)
            self.x = np.delete(self.x, np.arange(min(left_edge, 0), max(right_edge, len(self.x))))
            self.y = np.delete(self.y, np.arange(min(left_edge, 0), max(right_edge, len(self.y))))
            self.y_err = np.delete(self.y_err, np.arange(min(left_edge, 0), max(right_edge, len(self.y_err))))

            self.x_tensor = torch.tensor(self.x, dtype=torch.float32).to(self.device)
            self.y_tensor = torch.tensor(self.y, dtype=torch.float32).to(self.device)

            # Update num_detected_anomalies, flagged_anomalous, and anomalous_signal with the new flagged anomaly from left_edge to right_edge
            if neg_anomaly_only:
                # Check if the average of the residuals in the flagged region is negative
                if np.mean(self.y_orig[left_edge:right_edge] - pred_full_x[left_edge:right_edge]) <= 0:
                    self.num_detected_anomalies += 1
                    self.flagged_anomalous[left_edge:right_edge] = 1
                    self.anomalous_signal[left_edge:right_edge] = minimum_filter1d(np.abs(self.y_orig[left_edge:right_edge] - pred_full_x[left_edge:right_edge]), size=self.len_deviant, mode="nearest")
                    print(f"Anomalous edges = {left_edge}:{right_edge}")
                else:
                    print(f"Not flagging edges {left_edge}:{right_edge} because not a negative anomaly (mean(truth - pred) = {np.mean(self.y_orig[left_edge:right_edge] - pred_full_x[left_edge:right_edge])}), and neg_anomaly_only is {neg_anomaly_only}. Still will remove edges from GP fit.")
            
            elif pos_anomaly_only:
                # Check if the average of the residuals in the flagged region is positive
                if np.mean(self.y_orig[left_edge:right_edge] - pred_full_x[left_edge:right_edge]) >= 0:
                    self.num_detected_anomalies += 1
                    self.flagged_anomalous[left_edge:right_edge] = 1
                    self.anomalous_signal[left_edge:right_edge] = minimum_filter1d(np.abs(self.y_orig[left_edge:right_edge] - pred_full_x[left_edge:right_edge]), size=self.len_deviant, mode="nearest")
                    print(f"Anomalous edges = {left_edge}:{right_edge}")
                else:
                    print(f"Not flagging edges {left_edge}:{right_edge} because not a positive anomaly (mean(truth - pred) = {np.mean(self.y_orig[left_edge:right_edge] - pred_full_x[left_edge:right_edge])}), and pos_anomaly_only is {pos_anomaly_only}. Still will remove edges from GP fit.")

            else:
                # Flag as anomalous regardless of whether it's a positive or negative anomaly
                self.num_detected_anomalies += 1
                self.flagged_anomalous[left_edge:right_edge] = 1
                self.anomalous_signal[left_edge:right_edge] = minimum_filter1d(np.abs(self.y_orig[left_edge:right_edge] - pred_full_x[left_edge:right_edge]), size=self.len_deviant, mode="nearest")
                print(f"Anomalous edges = {left_edge}:{right_edge}")

            # Predict on reduced x_tensor to get new residuals for threshold checking
            model.eval()
            likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = model(self.x_tensor)

            residuals = np.abs(self.y - observed_pred.mean.cpu().numpy())

            if update_threshold:
                # Calculate threshold
                residual_var = np.var(residuals)
                sum_variances = self.y_err**2 + residual_var
                self.threshold = self.num_sigma_threshold * np.sqrt(sum_variances)
            else:
                # Remove points between left_edge and right_edge from the threshold
                self.threshold = np.delete(self.threshold, np.arange(min(left_edge, 0), max(right_edge, len(self.threshold))))

            # Compute the minimum value in each window of size len_deviant in the residuals array; check if there are still points above the threshold
            min_values = minimum_filter1d(residuals, size=self.len_deviant, mode="nearest")
            exist_points_above_threshold = np.any(min_values > self.threshold)

        self.runtime = time.time() - start_time
        print(f"Time taken for anomaly detection: {self.runtime} seconds")