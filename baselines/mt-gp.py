import gpytorch


class ICMExactGP(gpytorch.models.ExactGP):
    """Exact GP with an ARD RBF input kernel and rank-2 task covariance.

    Task IDs are integer-valued and supplied as a column tensor.
    The original wind-energy baseline uses six tasks and five input features.
    """

    def __init__(self, train_x, train_task, train_y, likelihood):
        super().__init__((train_x, train_task), train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.x_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=5)
        )
        self.task_kernel = gpytorch.kernels.IndexKernel(num_tasks=6, rank=2)

    def forward(self, x, task):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.x_kernel(x) * self.task_kernel(task)
        )
