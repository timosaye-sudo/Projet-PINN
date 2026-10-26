import torch
from torch.func import grad


def sortie(x: torch.Tensor, net, params, buffers) -> torch.Tensor:
    return torch.func.functional_call(net, (params, buffers), x.unsqueeze(-1)).squeeze()


# dérivées première et seconde par rapport à x
derivee = grad(sortie, argnums=0)
derivee_seconde = grad(derivee, argnums=0)


if __name__ == "__main__":
    from .models import SNN

    # la dérivée doit être proche de la différence finie
    torch.manual_seed(0)
    net = SNN(3)
    params = dict(net.named_parameters())
    buffers = dict(net.named_buffers())
    x, h = torch.tensor(0.3), 1e-3
    approx = (
        sortie(x + h, net, params, buffers) - sortie(x - h, net, params, buffers)
    ) / (2 * h)
    print(derivee(x, net, params, buffers).item(), approx.item())
