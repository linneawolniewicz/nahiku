import gc
import time
import torch
import gpytorch
import warnings
import numpy as np
import matplotlib.pyplot as plt

from nahiku.src.Search import Search
from nahiku.src.exhaustive_helpers import precompute_precision, interval_posterior_from_precision, compute_interval_pvalue

from gpytorch.mlls import ExactMarginalLogLikelihood


class ExhaustiveSearch(Search):
    """
        This class implements an exhaustive search algorithm to identify anomalous intervals in a time series using Gaussian Processes.

        Method:
            1. List every possible contiguous interval in the time series that could contain an anomaly, based on priors (e.g., minimum and maximum anoamly duration).
            2. Fit the entire time series with a GP and store the optimized parameters (and, optionally, the full precision matrix if using dynamic programming).
            3. For each candidate interval, compute the posterior likelihood of the test interval given a MultivariateNormal distribution fit to the rest of the data.
               Optionally, use dynamic programming to compute the posterior likelihoods more efficiently by leveraging the precision matrix of the full GP fit.
            4. Flag intervals as anomalous if a metric measuring posterior likelihood is below a certain threshold (e.g. mahalanobis distance below some threshold)
    """
    def __init__(
        self,
        x,                 # Map to Search.x
        y,                 # Map to Search.y
        dominant_period,   # Map to Search.dominant_period
        device="cpu",      # Map to Search.device
        min_anomaly_len=1,
        max_anomaly_len=400,
        window_slide_step=1,
        window_size_step=1,
        assume_independent=True,
        which_test_metric="mll",
    ):
        """
        Initialize the ExhaustiveSearch class and the base Search class with the provided parameters.
        
        :param x (np.ndarray): x array of the light curve
        :param y (np.ndarray): y array of the light curve
        :param dominant_period (float): dominant period of the light curve
        :param device (str): device to use for GP modeling (default: "cpu")
        :param min_anomaly_len (int): minimum length of candidate anomalous intervals (default: 1)
        :param max_anomaly_len (int): maximum length of candidate anomalous intervals (default: 400)
        :param window_slide_step (int): step size for sliding the window across the time series (default: 1)
        :param window_size_step (int): step size for varying the size of the candidate intervals (default: 1)
        :param assume_independent (bool): if True, assumes independence between points for speed. False is not yet implemented and will be ignored for now. (default: True)
        :param which_test_metric (str): metric to use for evaluating the likelihood of test intervals. 
               Options are 'pval', 'mahalanobis', 'nlpd', 'msll', 'rmse', 'mll', or default is 'll' (log-likelihood)
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

        # Initialize parameters
        self.min_anomaly_len = min_anomaly_len
        self.max_anomaly_len = max_anomaly_len
        self.window_slide_step = window_slide_step
        self.window_size_step = window_size_step
        self.assume_independent = assume_independent
        self.which_test_metric = which_test_metric
        self.num_steps = len(x)

        # Check that min_anomaly_len is at least 1 and that max_anomaly_len is at least min_anomaly_len
        if min_anomaly_len < 1:
            warnings.warn("min_anomaly_len must be at least 1. Setting min_anomaly_len to 1.")
            self.min_anomaly_len = 1
        if max_anomaly_len < min_anomaly_len:
            warnings.warn("max_anomaly_len must be at least min_anomaly_len. Setting max_anomaly_len to 10 x min_anomaly_len.")
            self.max_anomaly_len = 10 * self.min_anomaly_len

        # Possible candidate intervals
        self.intervals = []
        for start in range(0, self.num_steps - min_anomaly_len, window_slide_step):
            for end in range(
                start + min_anomaly_len,
                min(start + max_anomaly_len, self.num_steps),
                window_size_step,
            ):
                self.intervals.append((start, end))

        # Initialize variables to store results
        self.metrics = []
        self.pos_or_neg_intervals = []

    def search_for_anomaly(
        self,
        filename="",
        refit=False,
        neg_anomaly_only=False,
        pos_anomaly_only=False, 
        dynamic_programming=False, 
        threshold=1e-5,
        num_intervals_to_flag=None,
        silent=True,
        plot=False,
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
        Main function to perform the exhaustive search for anomalies in the time series data.
        
        :param filename (str): If provided, saves the results to this file (default: "")
        :param refit (bool): If true, refit the GP for each interval. If false, use the same GP for all intervals (faster but less accurate) (default: False)
        :param neg_anomaly_only (bool): Whether to only flag negative anomalies (i.e., dips) instead of both positive and negative anomalies (default: False)
        :param pos_anomaly_only (bool): Whether to only flag positive anomalies (i.e., flares) instead of both positive and negative anomalies (default: False)
        :param dynamic_programming (bool): If true, use dynamic programming to find the best interval. Only works if refit = False (default: False)
        :param threshold (float): Threshold for flagging an interval as anomalous based on the test metric (default: 1e-5)
        :param num_intervals_to_flag (int or None): If not None, flag the top num_intervals_to_flag intervals as anomalous based on the test metric, instead of using a threshold (default: None)
        :param silent (bool): If true, suppresses print statements during training (default: True)
        :param plot (bool): If true, plots the GP prediction and p-value for each candidate interval (default: False)
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

        # Check that both threshold and num_intervals_to_flag are not provided at the same time
        if threshold is not None and num_intervals_to_flag is not None:
            warnings.warn("Both threshold and num_intervals_to_flag are provided. Only one can be used. Setting num_intervals_to_flag to None and using threshold for flagging anomalous intervals.")
            num_intervals_to_flag = None

        # Check that if dynamic_programming is True, then refit must be False
        if dynamic_programming and refit:
            warnings.warn("Dynamic programming only works if refit is False. Setting refit to False.")
            refit = False

        # Check if dynamic_programming is True, then which_test_metric must be "pval" or "mahalanobis"
        if dynamic_programming and self.which_test_metric not in ["pval", "mahalanobis"]:
            warnings.warn("Dynamic programming only works with pval or mahalanobis metrics. Setting self.which_test_metric to 'pval'.")
            self.which_test_metric = "pval"

        # Initialize
        self.min_metric = np.inf
        self.best_interval = None

        # Write metrics to txt file if filename is provided
        if filename == "":
            save_to_txt = False
        else:
            # Create txt file to save results
            save_to_txt = True

            # write header
            with open(filename, "w") as f:
                f.write("start,end,metric\n")

        # Initialize kernel, likelihood, and mean based on self.dominant_period
        init_kernel = self.build_kernel()
        init_mean = self.build_mean()
        init_likelihood = self.build_likelihood()

        # If not refitting at each iteration, fit the GP to the entire data once and save the kernel parameters
        if not refit:
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

            # If using dynamic programming, precompute and cache the precision matrix for the full dataset
            if dynamic_programming:
                mu_full, J_full = precompute_precision(
                    full_x = self.x_tensor, 
                    mean_module = mean,
                    kernel_module = kernel,
                    noise_variance = model.likelihood.noise.item(),
                    dtype = torch.float64,  # or float32
                    device = self.device,
                )

        # Iterate over each possible anomaly interval
        for start, end in self.intervals:
            # Create train and test masks
            mask_train = np.ones(self.num_steps, dtype=bool)
            mask_train[start:end] = False
            mask_test = ~mask_train

            # Create train data without interval
            x_train = torch.tensor(self.x[mask_train], dtype=torch.float32).to(self.device)
            y_train = torch.tensor(self.y[mask_train], dtype=torch.float32).to(self.device)
            
            # Create test data with interval
            x_test = torch.tensor(self.x[mask_test], dtype=torch.float32).to(self.device)
            y_test = torch.tensor(self.y[mask_test], dtype=torch.float32).to(self.device)

            # If refitting at each iteration, fit the GP to the training data without the interval
            if refit:
                if 'model' in locals():
                    del likelihood

                model, likelihood, _, _ = self.build_gp_model(x_train, y_train, kernel=init_kernel, mean=init_mean, likelihood=init_likelihood, device=self.device) # Note: initializing from previous hyperparameters

                # Train GP on training data
                model, likelihood, _ = self.train_gp(
                    gp_model=model,
                    likelihood=likelihood,
                    x=x_train,
                    y=y_train,
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

                # Evaluate metric for prediction on test data
                model.eval()
                likelihood.eval()

                # Compute p-value for the interval
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    f_pred = model(x_test)
                    y_pred = likelihood(f_pred)

            else:
                if dynamic_programming:
                    # Use precomputed precision to get posterior on test interval
                    y_pred = interval_posterior_from_precision(
                        mu = mu_full,
                        J = J_full,
                        full_y = self.y_tensor,
                        mask_train = mask_train,
                        mask_test = mask_test,
                        dtype = J_full.dtype, 
                    )

                else:
                    # Don't refit; but create a new GP model over train data with previously optimized parameters
                    model, likelihood, _, _ = self.build_gp_model(x_train, y_train, kernel=kernel, mean=mean, likelihood=likelihood, device=self.device) # Note: initializing from previous hyperparameters
                    
                    # Evaluate metric for prediction on test data
                    model.eval()
                    likelihood.eval()

                    # Compute p-value for the interval
                    with torch.no_grad(), gpytorch.settings.fast_pred_var():
                        f_pred = model(x_test)
                        y_pred = likelihood(f_pred)

            # Store as positive or negative interval based on whether the mean prediction is above or below the observed values in the test interval
            if np.mean(y_test.cpu().numpy() - y_pred.mean.cpu().numpy()) <= 0:
                self.pos_or_neg_intervals.append(-1)
            else:
                self.pos_or_neg_intervals.append(1)

            # Depending on which_test_metric, compute the appropriate metric to evaluate the likelihood of the test interval under the model trained on the train interval
            if self.which_test_metric == "pval" or self.which_test_metric == "mahalanobis":
                maha_dist, p_value = compute_interval_pvalue(y_test, y_pred)

                if self.which_test_metric == "pval":
                    interval_metric = p_value
                else:
                    interval_metric = maha_dist

            else:
                # These metrics are computed pointwise, so we compute them for each point in the interval and then average them to get a single metric for the interval
                metric_sum = 0
                for i in range(end - start):
                    # For each point in the interval, calculate the metric and sum them up
                    x_curr = x_test[i].unsqueeze(0)
                    y_curr = y_test[i].unsqueeze(0)

                    with torch.no_grad(), gpytorch.settings.fast_pred_var():
                        f_pred = model(x_curr)
                        y_pred = likelihood(f_pred)

                    if self.which_test_metric == "nlpd":
                        metric = gpytorch.metrics.negative_log_predictive_density(
                            y_pred, y_curr
                        )

                    elif self.which_test_metric == "msll":
                        metric = gpytorch.metrics.mean_standardized_log_loss(y_pred, y_curr)

                    elif self.which_test_metric == "rmse":
                        pred_mean = y_pred.mean.cpu().numpy()
                        metric = np.sqrt(np.mean((pred_mean - y_curr.cpu().numpy()) ** 2))

                    elif self.which_test_metric == "mll":
                        mll_func = ExactMarginalLogLikelihood(likelihood, model)
                        metric = mll_func(f_pred, y_curr)

                    else:  # Default to log-likelihood
                        metric = y_pred.log_prob(y_curr)

                    metric_sum += metric

                # Calculate the mean of the metric over the interval
                interval_metric = metric_sum / (end - start)

            # Save the metric for the interval
            self.metrics.append(interval_metric)

            # Check if the current interval is the best one
            if interval_metric < self.min_metric:
                self.min_metric = interval_metric
                self.best_interval = (start, end)

            # Print results for the interval if not silent
            if not silent:
                print(
                    f"Anomaly interval: {start}-{end}, metric {self.which_test_metric} over the interval: {interval_metric}, pos or neg: {self.pos_or_neg_intervals[-1]}"
                )

            if plot:
                if dynamic_programming:
                    # Create a train GP model just to get the kernel and mean for plotting; but we won't use it for inference
                    model = model, likelihood, _, _ = self.build_gp_model(x_train, y_train, kernel=kernel, mean=mean, likelihood=likelihood, device=self.device)

                    # Evaluate metric for prediction on test data
                    model.eval()
                    likelihood.eval()

                # Compute predictions for plotting
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    f_pred = model(self.x_tensor)
                    y_pred = likelihood(f_pred)
                    pred_mean = y_pred.mean.cpu().numpy()
                    one_stdev = y_pred.stddev.cpu().numpy()

                # Plot the results
                plt.figure(figsize=(7, 5))
                plt.title(f"p-value: {interval_metric:.0e}")
                plt.fill_between(
                    self.x, pred_mean - one_stdev, pred_mean + one_stdev, alpha=0.5
                )
                plt.plot(self.x, pred_mean, lw=2, alpha=0.9, label="Predicted $\pm$ 1 $\sigma$")
                plt.plot(self.x, self.y, "k.", markersize=3, label="Observed")
                plt.plot(
                    self.x[start:end], self.y[start:end], "r.", markersize=4, label="Held Out Interval"
                )

                plt.xlabel("Time [days]")
                plt.ylabel("Standardized Flux")
                plt.xlim(min(self.x), max(self.x))
                plt.legend()

                plt.show()

            # Save results to txt if save_to_txt is True
            if save_to_txt:
                with open(filename, "a") as f:
                    f.write(f"{start},{end},{interval_metric}\n")

            # Delete all variables to free up memory
            if 'model' in locals():
                del model
            
            if 'f_pred' in locals():
                del f_pred

            del x_train
            del y_train
            del y_pred
            del interval_metric
            del mask_train
            del mask_test
            del x_test
            del y_test
            gc.collect()
            torch.cuda.empty_cache()


        # After iterating through all intervals, update self.num_detected_anomalies, self.flagged_anomalous, and self.anomalous_signal based on the intervals found
        if threshold is not None:
            # Flag intervals as anomalous if their metric is below the threshold
            for i, metric in enumerate(self.metrics):
                if metric < threshold:
                    start, end = self.intervals[i]
                    self.flagged_anomalous[start:end] = True
                    self.anomalous_signal[start:end] = metric
                    self.num_detected_anomalies += 1

        elif num_intervals_to_flag is not None:            
            # Flag the top num_intervals_to_flag intervals as anomalous based on the metric
            sorted_indices = np.argsort(self.metrics)
            for i in range(min(num_intervals_to_flag, len(self.intervals))):
                idx = sorted_indices[i]
                start, end = self.intervals[idx]
                self.flagged_anomalous[start:end] = True
                self.anomalous_signal[start:end] = self.metrics[idx]
                self.num_detected_anomalies += 1

        self.runtime = time.time() - start_time
        print(f"Time taken for anomaly detection: {self.runtime} seconds")