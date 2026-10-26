import torch


def f(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(2 * torch.pi * x)


# solution exacte de -u'' = f avec u(0) = u(1) = 0
def solution_poisson(x: torch.Tensor) -> torch.Tensor:
    return f(x) / (4 * (torch.pi**2))


# solution exacte de -u'' + u = f avec u(0) = u(1) = 0
def solution_helmholtz(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(2 * torch.pi * x) / (1 + 4 * (torch.pi**2))


if __name__ == "__main__":
    # les deux écarts doivent être proches de 0 (différences finies)
    x = torch.linspace(0.1, 0.9, 9, dtype=torch.float64)
    h = 1e-4
    u = solution_poisson(x)
    u_xx = (solution_poisson(x + h) - 2 * u + solution_poisson(x - h)) / h**2
    print("poisson :", (-u_xx - f(x)).abs().max().item())
    u = solution_helmholtz(x)
    u_xx = (solution_helmholtz(x + h) - 2 * u + solution_helmholtz(x - h)) / h**2
    print("helmholtz :", (-u_xx + u - f(x)).abs().max().item())
