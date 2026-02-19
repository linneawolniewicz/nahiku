import torch
import gpytorch
import warnings
import numpy as np
import matplotlib.pyplot as plt

from nahiku.src.kernels import QuasiPeriodicKernel, ExactGPModel

from abc import abstractmethod
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import PeriodicKernel, RBFKernel, ScaleKernel
from gpytorch.constraints import Interval, GreaterThan

class Search:
    """
    Base class for anomaly detection in light curves using Gaussian Processes.
    This base class provides common functionality for different search strategies (e.g., greedy, exhaustive) and build methods
    for constructing and optimizing a Gaussian Process model with a quasi-periodic kernel on the x and y data of a light curve.
    """

    def __init__(
        self,
        x,
        y,
        dominant_period,
        device="cpu",
    ):
        """
        Build the Search object with x, y, and optional parameters for GP modeling.
        
        :param x (np.ndarray): x array of the light curve
        :param y (np.ndarray): y array of the light curve
        :param dominant_period (float): dominant period of the light curve
        :param device (str): device to use for GP modeling (default: "cpu")
        """

        self.x = x
        self.y = y
        self.x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        self.y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
        self.dominant_period = dominant_period
        self.device = device

        # Initialize variables to store detected anomalies and runtime
        self.num_detected_anomalies = 0
        self.flagged_anomalous = np.zeros_like(self.x, dtype=bool)
        self.anomalous_signal = np.zeros_like(self.x)
        self.runtime = 0

    def build_kernel(self):
        """
        Build the kernel for the GP model based on the dominant period of the light curve.
        Creates a Quasi-Periodic Kernel with constraints on the period length and lengthscales to guide the optimization process, 
        and wraps it in a ScaleKernel to allow for scaling of the overall kernel output.
        """

        # Define kernel initial values based on the dominant period scale
        if self.dominant_period <= 0.5:
            period_length_constraint = Interval(
                lower_bound=max(0 + 1e-3, self.dominant_period - 0.1), upper_bound=2, initial_value=self.dominant_period
            )
            periodic_lengthscale_constraint = GreaterThan(
                lower_bound=self.dominant_period/4, initial_value=self.dominant_period
            )
            rbf_lengthscale_constraint = GreaterThan(
                lower_bound=self.dominant_period/4, initial_value=self.dominant_period * 4
            )

        elif self.dominant_period >= 0.5 and self.dominant_period < 1:
            period_length_constraint = Interval(
                lower_bound=0.4, upper_bound=5, initial_value=self.dominant_period
            )
            periodic_lengthscale_constraint = GreaterThan(
                lower_bound=0.1, initial_value=self.dominant_period
            )
            rbf_lengthscale_constraint = GreaterThan(
                lower_bound=1, initial_value=self.dominant_period * 3
            )

        elif self.dominant_period >= 1 and self.dominant_period < 4:
            period_length_constraint = Interval(
                lower_bound=0.4, upper_bound=5, initial_value=self.dominant_period
            )
            periodic_lengthscale_constraint = GreaterThan(
                lower_bound=0.2, initial_value=self.dominant_period / 2
            )
            rbf_lengthscale_constraint = GreaterThan(
                lower_bound=1, initial_value=self.dominant_period * 2
            )

        elif self.dominant_period >= 4 and self.dominant_period < 8:
            period_length_constraint = Interval(
                lower_bound=self.dominant_period - 1,
                upper_bound=self.dominant_period + 1,
                initial_value=self.dominant_period,
            )
            periodic_lengthscale_constraint = GreaterThan(
                lower_bound=0.4, initial_value=self.dominant_period / 4
            )
            rbf_lengthscale_constraint = GreaterThan(
                lower_bound=self.dominant_period / 3, initial_value=self.dominant_period * 1.5
            )

        else:
            period_length_constraint = Interval(
                lower_bound=self.dominant_period - 2,
                upper_bound=self.dominant_period + 2,
                initial_value=self.dominant_period,
            )
            periodic_lengthscale_constraint = GreaterThan(lower_bound=0.4, initial_value=2)
            rbf_lengthscale_constraint = GreaterThan(
                lower_bound=self.dominant_period / 4, initial_value=self.dominant_period
            )

        # Define the GP model
        qp_kernel = QuasiPeriodicKernel(
            periodic_kernel=PeriodicKernel(
                period_length_constraint=period_length_constraint,
                lengthscale_constraint=periodic_lengthscale_constraint,
            ),
            rbf_kernel=RBFKernel(lengthscale_constraint=rbf_lengthscale_constraint),
        )

        # Wrap the kernel in a ScaleKernel to allow for scaling of the overall kernel output
        kernel = ScaleKernel(
            qp_kernel, outputscale_constraint=Interval(lower_bound=0, upper_bound=5, initial_value=1)
        ).to(self.device)

        return kernel
    
    def build_likelihood(self):
        """
        Build the likelihood for the GP model.
        """

        # Define likelihood with a constraint on the noise level to prevent it from becoming too small during optimization
        likelihood = GaussianLikelihood(
            noise_constraint=GreaterThan(lower_bound=1e-4, initial_value=0.5)
        ).to(self.device)

        return likelihood

    def build_mean(self):
        """
        Build the mean function for the GP model.
        """

        # Define mean function with a constraint on the constant value to prevent it from deviating too far from 0 during optimization 
        # (since data is assumed standardized)
        mean = ConstantMean(constant_constraint=Interval(lower_bound=-1, upper_bound=1, initial_value=0.01)).to(self.device)

        return mean

    def build_gp_model(
        self,
        x=None,
        y=None,
        kernel=None,
        likelihood=None,
        mean=None,
        device=None,
    ):
        """
        Initialize an ExactGPModel with the defined kernel, likelihood, and mean function.

        :param kernel (GPytorch.kernel object): kernel to use for the GP model (optional)
        :param likelihood (GPytorch.likelihood object): likelihood to use for the GP model (optional)
        :param mean (GPytorch.mean object): mean function to use for the GP model (optional)
        """

        kernel = kernel if kernel is not None else self.build_kernel()
        likelihood = likelihood if likelihood is not None else self.build_likelihood()
        mean = mean if mean is not None else self.build_mean()
        train_x = x if x is not None else self.x_tensor
        train_y = y if y is not None else self.y_tensor
        device = device if device is not None else self.device

        gp = ExactGPModel(
            train_x=train_x, 
            train_y=train_y,
            kernel=kernel,
            likelihood=likelihood,
            mean=mean
        ).to(device)

        return gp, likelihood, kernel, mean

    def train_gp(
        self,
        gp_model,
        likelihood,
        training_iterations=1_000,
        lr=0.01,
        which_metric="mll",
        which_opt="adam",
        early_stopping=True,
        min_iterations=None,
        patience=1,
        plot=False,
        set_noise_equal_to_var_residuals=False,
        x=None,
        y=None,
        device=None,
    ):
        """
        Train the GP model using the specified parameters and return the trained model, likelihood, and final log likelihood value.

        :param training_iterations (int): maximum number of training iterations (default: 1000)
        :param lr (float): learning rate for the optimizer (default: 0.01)
        :param which_metric (str): Metric to use for evaluating improvement during training. Options are 'mll' for marginal log likelihood and 'mse' for mean squared error. Default is 'mll'.
        :param which_opt (str): Optimizer to use for training. Options are 'adam' and 'sgd'. Default is 'adam'.
        :param early_stopping (bool): Whether to use early stopping based on the training loss (default: True)
        :param min_iterations (int or None): Minimum number of iterations to train before considering early stopping (default: None, which sets it to training_iterations // 10)
        :param patience (int): Number of consecutive iterations with increasing loss to wait before stopping when early_stopping is True (default: 1)
        :param plot (bool): Whether to plot the training loss and covariance matrices after training (default: False)
        :param set_noise_equal_to_var_residuals (bool): Whether to set the likelihood noise variance equal to the variance of the residuals after training (default: False)
        :param x (torch.Tensor or None): Training input data (optional, defaults to self.x_tensor)
        :param y (torch.Tensor or None): Training target data (optional, defaults to self.y_tensor)
        :param device (str or None): Device to use for training (optional, defaults to self.device)
        """

        # Get training data and device if not provided
        x = x if x is not None else self.x
        y = y if y is not None else self.y
        device = device if device is not None else self.device
    
        # Validate which_metric input
        if which_metric not in ["mll", "mse"]:
            warnings.warn(f"Only 'mll' or 'mse' are supported values for which_metric, not {which_metric}. Choosing 'mll' by default.")
        
        # Set minimum iterations for early stopping if not provided
        if early_stopping and (min_iterations is None):
            min_iterations = training_iterations // 10
            print(f"Using {min_iterations} as minimum iterations for early stopping.")
        elif not early_stopping and min_iterations is not None:
            warnings.warn("min_iterations is set but early_stopping is False, so min_iterations will be ignored.")

        # Check training_iterations > 0
        if training_iterations <= 0:
            warnings.warn(f"training_iterations {training_iterations} must be a positive integer. Setting to 1000 by default.")
            training_iterations = 1000

        # Set model and likelihood into training mode
        gp_model.train()
        likelihood.train()

        # Set optimizer
        if which_opt == "adam":
            optimizer = torch.optim.Adam(gp_model.parameters(), lr=lr)
        elif which_opt == "sgd":
            optimizer = torch.optim.SGD(gp_model.parameters(), lr=lr)
        else:
            warnings.warn("which_opt must be either 'adam' or 'sgd'. Defaulting to 'adam'.")
            optimizer = torch.optim.Adam(gp_model.parameters(), lr=lr)

        # Set up the marginal log likelihood for optimization
        mll = ExactMarginalLogLikelihood(likelihood, gp_model)

        # Plot loss during training
        train_losses = []
        counter = 0
        min_loss = float("inf")

        # For each training iteration, zero gradients, compute output and loss, backpropagate, and step optimizer
        for i in range(training_iterations):
            optimizer.zero_grad()
            pred = gp_model(x)

            # Compute losses
            if which_metric == "mse":
                train_loss = torch.nn.functional.mse_loss(pred.mean, y)
            else:
                train_loss = -mll(pred, y)

            # Early stopping
            if early_stopping:
                if i > min_iterations:
                    # If loss decreased, reset counter
                    if train_loss < min_loss:
                        min_loss = train_loss
                        counter = 0

                    # If loss increased, increment counter and check if patience is exceeded
                    else:
                        counter += 1
                        if counter > patience:
                            print(
                                f"Early stopping at iteration {i} due to increasing train loss."
                            )
                            break

            train_loss.backward()
            optimizer.step()

            # Save losses
            train_losses.append(train_loss.item())
        
        # Compute final log likelihood value
        log_likelihood_value = mll(pred, y).item()

        # If set_noise_equal_to_var_residuals is True, set the likelihood noise parameter equal to the variance of the residuals
        if set_noise_equal_to_var_residuals:
            # Variance analysis
            gp_model.eval()
            likelihood.eval()

            old_noise = likelihood.noise_covar.noise.item()

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                # Predict on the train data
                predictions_train = likelihood(gp_model(x))
                mean_train = predictions_train.mean.cpu().numpy()
                calc_variance = max(1e-4, np.var(y.cpu().numpy() - mean_train))

            likelihood.noise = torch.tensor(calc_variance).to(device)
            likelihood.noise_covar.noise = torch.tensor(calc_variance).to(device)
            print(f"Setting learned noise variance {old_noise:.3f} to residual variance: {calc_variance:.3f}; likelihood noise variance = {likelihood.noise_covar.noise.item():.3f}")

        if plot:
            # Plot the train loss
            plt.figure(figsize=(20, 5))
            plt.subplot(1, 4, 1)
            plt.plot(range(len(train_losses)), train_losses)
            plt.xlabel("Iteration")
            plt.ylabel("Train Loss")
            plt.title("Train Loss for metric " + which_metric)

            # Plot the covariance matrices
            with torch.no_grad():
                periodic_cov = gp_model.covar_module.base_kernel.periodic_kernel(
                    torch.tensor(x).to(device)
                ).evaluate()

                rbf_cov = gp_model.covar_module.base_kernel.rbf_kernel(
                    torch.tensor(x).to(device)
                ).evaluate()

                cov_matrix = gp_model.covar_module(torch.tensor(x).to(device)).evaluate()

            plt.subplot(1, 4, 2)
            plt.title("Covariance Matrix")
            plt.imshow(cov_matrix.cpu().numpy(), cmap="viridis")
            plt.colorbar()

            plt.subplot(1, 4, 3)
            plt.title("Periodic Kernel Covariance")
            plt.imshow(periodic_cov.cpu().numpy(), cmap="viridis")
            plt.colorbar()

            plt.subplot(1, 4, 4)
            plt.title("RBF Kernel Covariance")
            plt.imshow(rbf_cov.cpu().numpy(), cmap="viridis")
            plt.colorbar()

        return gp_model, likelihood, log_likelihood_value
        
    @abstractmethod
    def search_for_anomaly(self):
        """All subclasses must implement this method."""
        pass



    