import gpytorch

from gpytorch.kernels import Kernel, PeriodicKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal

# Create a Quasi-Periodic Kernel
class QuasiPeriodicKernel(Kernel):
    """
    A custom kernel that combines a periodic kernel with an RBF kernel to model quasi-periodic behavior in light curves.
    """

    def __init__(
            self, 
            periodic_kernel=None, 
            rbf_kernel=None, 
            **kwargs
        ):

        """
        Initialize the QuasiPeriodicKernel with optional periodic and RBF kernels.
        
        :param periodic_kernel (GPyTorch.kernel object): An instance of a periodic kernel (optional)
        :param rbf_kernel (GPyTorch.kernel object): An instance of an RBF kernel (optional)
        """

        super(QuasiPeriodicKernel, self).__init__(**kwargs)

        # If no periodic kernel is provided, initialize a default one
        if periodic_kernel is None:
            self.periodic_kernel = PeriodicKernel(**kwargs)
        else:
            self.periodic_kernel = periodic_kernel

        # If no RBF kernel is provided, initialize a default one
        if rbf_kernel is None:
            self.rbf_kernel = RBFKernel(**kwargs)
        else:
            self.rbf_kernel = rbf_kernel

    def forward(
            self, 
            x1, 
            x2, 
            diag=False, 
            **params
        ):
        """
        Compute the kernel value between two sets of inputs by combining the periodic and RBF kernels.
        
        :param x1 (torch.Tensor): First set of input points
        :param x2 (torch.Tensor): Second set of input points
        :param diag (bool): Whether to compute only diagonal elements
        """
        periodic_part = self.periodic_kernel.forward(x1, x2, diag=diag, **params)
        rbf_part = self.rbf_kernel.forward(x1, x2, diag=diag, **params)
        
        return periodic_part * rbf_part


# Create a parameterized ExactGP model, initialized with training data x and y
class ExactGPModel(gpytorch.models.ExactGP):
    """
    Initialize the ExactGPModel with a specified kernel and mean function, and define the forward method to compute the GP output.
    """
    def __init__(self, train_x, train_y, likelihood, kernel, mean):
        """
        Initialize the ExactGPModel with training data, likelihood, kernel, and mean function.
        
        :param train_x (torch.Tensor): Training input data
        :param train_y (torch.Tensor): Training target data
        :param likelihood (GPyTorch.likelihood object): Likelihood function for the GP model
        :param kernel (GPyTorch.kernel object): Kernel function for the GP model
        :param mean (GPyTorch.mean object): Mean function for the GP model
        """

        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = mean
        self.covar_module = kernel
        self.likelihood = likelihood

    def forward(self, x):
        """
        Compute the forward pass of the ExactGPModel.
        
        :param x (torch.Tensor): Input data points
        """

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


# Create parameterized GP model, initialized without training data
class ParameterizedGPModel(gpytorch.models.GP):
    """
    Base GP model that allows for parameterized kernels and mean functions, without the need for training data at initialization.
    """
    def __init__(self, kernel, mean):
        """
        Initialize the ParameterizedGPModel with a specified kernel and mean function.
        
        :param kernel (GPyTorch.kernel object): Kernel function for the GP model
        :param mean (GPyTorch.mean object): Mean function for the GP model
        """

        super().__init__()
        self.mean_module = mean
        self.covar_module = kernel

    def forward(self, x):
        """
        Compute the forward pass of the ParameterizedGPModel.
        
        :param x (torch.Tensor): Input data points
        """

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)