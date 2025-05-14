import matplotlib.pyplot as plt
import torch
from einops import repeat
from matplotlib.axes import Axes
from tqdm.auto import tqdm

from pytorch_from_scratch.p03_optimizers import my_optimizers


def rosenbrocks_banana(x: torch.Tensor, y: torch.Tensor, a=1, b=100) -> torch.Tensor:
    return (a - x) ** 2 + b * (y - x**2) ** 2 + 1


def plot_rosenbrock(
    xmin=-2, xmax=2, ymin=-1, ymax=3, n_points=50, log_scale=False
) -> Axes:
    """Plot the rosenbrocks_banana function over the specified domain.

    If log_scale is True, take the logarithm of the output before plotting.
    """
    x = repeat(torch.linspace(xmin, xmax, n_points), "x -> y x", x=n_points, y=n_points)
    y = repeat(torch.linspace(ymax, ymin, n_points), "y -> y x", x=n_points, y=n_points)
    z = rosenbrocks_banana(x, y)
    if log_scale:
        z = z.log()
    fig, ax = plt.subplots()
    ax.imshow(z, extent=(xmin, xmax, ymin, ymax))
    ax.plot([1], [1], ".w")
    return ax


def optimize_rosenbrock(
    optim_f,
    optim_params,
    xy: torch.Tensor,
    n_iters: int,
):
    """Optimize the banana starting from the specified point.

    xy: shape (2,). The (x, y) starting point.
    n_iters: number of steps.

    Return: (n_iters, 2). The (x,y) BEFORE each step. So out[0] is the starting point.
    """
    assert xy.requires_grad
    points = xy.new_zeros((n_iters, 2))
    optim = optim_f([xy], **optim_params)
    for i in tqdm(range(n_iters)):
        points[i] = xy.detach()
        optim.zero_grad()
        loss = rosenbrocks_banana(xy[0], xy[1])
        loss.backward()
        optim.step()
    return points


def optimize_and_plot(optim_f, optim_params, xy=(-1.5, 2.5), n_iters=100, title=None):
    xy = torch.tensor(list(xy), requires_grad=True)
    xys = optimize_rosenbrock(optim_f, optim_params, xy, n_iters=n_iters).numpy()
    ax = plot_rosenbrock(log_scale=True)
    ax.plot(xys[:, 0], xys[:, 1], color="r", linewidth=1, marker="")
    ax.plot(xys[-1:, 0], xys[-1:, 1], color="y", linewidth=1, marker=".")
    ax.set_title(f"{title or ''}\nlast_point={xys[-1]}")
    return ax


def main():
    plot_rosenbrock().set_title("Rosenbrocks Banana")
    plot_rosenbrock(log_scale=True).set_title("Rosenbrocks Banana (log scale)")
    optimize_and_plot(
        optim_f=torch.optim.SGD,
        optim_params=dict(lr=0.001, momentum=0.98),
        title="torch.optim.SGD",
    )
    optimize_and_plot(
        optim_f=my_optimizers.SGD,
        optim_params=dict(lr=0.001, momentum=0.98 / 1.1, weight_decay=0.15),
        n_iters=1000,
        title="my_optimizers.SGD",
    )
    optimize_and_plot(
        optim_f=my_optimizers.RMSprop,
        optim_params=dict(
            lr=0.001,
            momentum=(1 - 0.02) / 1.3,
            weight_decay=0.3 / 2,
            alpha=0.3,
            eps=1e-5,
        ),
        n_iters=1000,
        title="my_optimizers.RMSprop",
    )
    optimize_and_plot(
        optim_f=my_optimizers.Adam,
        optim_params=dict(lr=0.001, weight_decay=0.3, betas=(0.2, 0.2), eps=1e-5),
        n_iters=1000,
        title="my_optimizers.Adam",
    )
    plt.show()


if __name__ == "__main__":
    main()
