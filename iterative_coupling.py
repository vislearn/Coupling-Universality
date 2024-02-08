import inspect
import os
import pickle
from argparse import ArgumentParser
from pathlib import Path
from time import time

import numpy as np
import torch
import yaml
from scipy.interpolate import interp1d
from PIL import Image
from matplotlib import pyplot as plt
from tqdm.auto import trange, tqdm

import density_push as dp


def train(dens0, data_len, n_steps, angle_mode, spline_region_count, step_size,
          spline_kind, fill_mode, resample_every, oas_steps):
    rots = []
    layers = []
    losses = []
    if isinstance(step_size, (float, int)):
        step_sizes = [step_size] * n_steps
    else:
        step_sizes = step_size
    data = dens0.sample_from(data_len)
    logp = 0
    with trange(n_steps - len(layers)) as pbar:
        for idx in pbar:
            if resample_every > 0 and idx > 0 and idx % resample_every == resample_every:
                data = dens0.sample_from(data_len)
                logp = 0
                for rot, layer in zip(rots, layers):
                    data, dlog = layer(rot.direct(data))
                    logp = logp + dlog

            # Find best rotation by trying each variant (OAS from Draxler et al. GCPR 2020)
            oas_layers = []
            for oas_step in range(oas_steps):
                if angle_mode.startswith("inc_"):
                    assert oas_steps == 1
                    angle = idx / float(angle_mode[angle_mode.index("_") + 1])
                elif angle_mode.startswith("add_"):
                    assert oas_steps == 1
                    angle = float(angle_mode[angle_mode.index("_") + 1])
                elif angle_mode == "rand":
                    angle = torch.rand(1) * 2 * np.pi
                else:
                    raise ValueError(f"Angle mode {angle_mode} not known.")
                rot_new = dp.Rotate(angle)
                data_rot = rot_new.direct(data)
                train_data = data_rot[torch.randperm(data_len)[:data_len // 4]]
                layer_new = layer_from_data(
                    train_data,
                    train_data.shape[0] // spline_region_count,
                    step_sizes[idx],
                    spline_kind,
                    fill_mode
                )

                data_new, dlog_new = layer_new(data_rot)
                logp_new = logp + dlog_new
                loss_new = ((data_new ** 2).mean() / 2 - logp_new.mean() / dens0.dim).item()

                oas_layers.append((loss_new, rot_new, layer_new, logp_new, data_new))

            loss, rot, layer, logp, data = min(oas_layers, key=lambda x: x[0])
            losses.append(loss)
            # if ex is not None:
            #     ex.log_scalar("train_loss", float(loss))
            pbar.set_description(f"{loss:.5f}")
            # Apply layer
            rots.append(rot)
            layers.append(layer)
    return layers, rots


def validate_loss(exp_dir, dens0, layers, rots):
    data = dens0.sample_from(2 ** 15)
    losses = []
    j = torch.tensor(0.)
    with torch.no_grad():
        for rot, layer in zip(tqdm([None, *rots]), [None, *layers]):
            if layer is not None:
                dj = layer.log_det_jacobian(rot.direct(data))
                data = layer.direct(rot.direct(data))
                j = j + dj

            losses.append(
                (data ** 2).mean(-1) / 2 - j.mean() / dens0.dim
            )
    kls = np.array(losses) + float((dens0.entropy(cached=False) + dp.GaussianDensity().log_normalization) / 2)

    plt.figure()

    mean = kls.mean(-1)
    std = kls.std(-1)
    plt.plot(mean)

    err = std / np.sqrt(kls.shape[-1])
    plt.fill_between(np.arange(kls.shape[0]), mean - err, mean + err, alpha=0.3)
    plt.axhline(0, color="black", linewidth=1)
    plt.legend()
    plt.xlabel("Layer")
    plt.ylabel("KL divergence $KL(p_\\theta(z)\\|p(z))$")
    # plt.xscale("log")
    # plt.yscale("log")  # "symlog", linthresh=np.min(shifted_losses, axis=0)[1])
    plt.savefig(exp_dir / "val_loss.pdf")
    plt.close()
    return sum(kls[-1]) / len(kls[-1])


def visualize_densities(exp_dir, dens0, layers, rots):
    transport_centered = centered_chain(rots, layers)
    transport_centered.pbar = True
    latent_estimate = dp.PushForwardDensity(dens0, transport_centered)

    grid = dp.vis.density_mesh(latent_estimate, -3, 3, 1000,
                               mesh_mode=dp.vis.MESH_MODE_RETURN_GRID)
    img = dp.vis.value_grid_to_image(grid)
    img = Image.fromarray((img * 255).astype(np.uint8), mode="RGBA")
    img.save(exp_dir / "latent_estimate.png")

    data_estimate = dp.PullBackwardDensity(dp.GaussianDensity(),
                                           transport_centered)

    grid = dp.vis.density_mesh(data_estimate, -3, 3, 1000,
                               mesh_mode=dp.vis.MESH_MODE_RETURN_GRID)
    img = dp.vis.value_grid_to_image(grid)
    img = Image.fromarray((img * 255).astype(np.uint8), mode="RGBA")
    img.save(exp_dir / "data_estimate.png")


class s_t_wrapper:
    def __init__(self, s_spline, t_spline):
        self.s_spline = s_spline
        self.t_spline = t_spline

    def __call__(self, pos):
        return (
            torch.from_numpy(self.s_spline(pos.numpy())),
            torch.from_numpy(self.t_spline(pos.numpy()))
        )


class const_spline():
    def __init__(self, knots, values, fill_value):
        self.knots = knots
        self.values = values
        self.fill_value = fill_value

    def __call__(self, positions):
        condlist = [
            positions <= knot
            for knot in self.knots
        ]
        choicelist = self.values
        return np.select(condlist, choicelist, default=self.fill_value)


class ScikitRegression():
    def __init__(self, sk_class, knots, values, **kwargs):
        self.model = sk_class(**kwargs)
        self.model.fit(knots[:, None].numpy(), values.numpy())

    def __call__(self, positions):
        return self.model.predict(positions[:, None])


def convolve(inp, weight):
    return torch.from_numpy(np.convolve(inp, weight, mode="same"))


def estimate_conditional_moments(data, count_per_bin):
    passive_data = data[:, 0]
    active_data = data[:, 1]

    sorted_passive = passive_data.sort()
    sorted_active = active_data[sorted_passive.indices]

    # This turned out to work less well than the simple mean/std approach
    # conv_weight = torch.ones(count_per_bin) / count_per_bin
    # conditional_mus = convolve(sorted_active, conv_weight)
    # conditional_sqrd = convolve(sorted_active ** 2, conv_weight)
    # conditional_pos = sorted_passive.values
    # conditional_stds = (conditional_sqrd - conditional_mus ** 2).sqrt()
    conditional_mus = sorted_active.reshape(-1, count_per_bin).mean(-1)
    conditional_stds = sorted_active.reshape(-1, count_per_bin).std(-1, unbiased=False)
    conditional_pos = sorted_passive.values.reshape(-1, count_per_bin).mean(-1)
    return conditional_pos, conditional_mus, conditional_stds


def layer_from_data(data, count_per_bin, step_size, spline_kind, fill_mode):
    conditional_pos, conditional_mus, conditional_stds = estimate_conditional_moments(
        data, count_per_bin)

    tgt_s = 1 / conditional_stds
    tgt_t = -conditional_mus / conditional_stds

    tgt_s = (1 + step_size * (tgt_s - 1))
    tgt_t = step_size * tgt_t

    if fill_mode == "identity":
        fill_value_s = 1
        fill_value_t = 0
    elif fill_mode == "extrapolate-const":
        fill_value_s = (tgt_s[0], tgt_s[-1])
        fill_value_t = (tgt_t[0], tgt_t[-1])
    else:
        raise ValueError(f"Fill mode {fill_mode} not known.")
    s_spline = interp1d(conditional_pos, tgt_s, kind=spline_kind,
                        bounds_error=False, fill_value=fill_value_s,
                        assume_sorted=True)
    t_spline = interp1d(conditional_pos, tgt_t, kind=spline_kind,
                        bounds_error=False, fill_value=fill_value_t,
                        assume_sorted=True)
    # s_spline = const_spline(conditional_rights.numpy(), tgt_s.numpy(), fill_value=1)
    # t_spline = const_spline(conditional_rights.numpy(), tgt_t.numpy(), fill_value=0)
    # s_spline = ScikitRegression(DecisionTreeRegressor, conditional_pos, tgt_s)  # , kernel="rbf")
    # t_spline = ScikitRegression(DecisionTreeRegressor, conditional_pos, tgt_t)  # , kernel="rbf")

    return dp.CouplingAffineTransport(s_t_wrapper(s_spline, t_spline))


def centered_chain(rots, layers):
    layer_chain = []
    acc_angle = 0
    for rot, layer in zip(rots, layers):
        acc_angle = acc_angle + rot.angle
        layer_chain.append(
            dp.TransportMapChain([
                dp.Rotate(acc_angle),
                layer,
                dp.Rotate(-acc_angle)
            ])
        )
    return dp.TransportMapChain(layer_chain)


def save_transport(exp_dir, layers, rots):
    with open(exp_dir / "model.p", "wb") as file:
        pickle.dump({
            "rots": rots,
            "layers": layers,
        }, file)


def main(
        exp_dir: Path,
        density: str = dp.TOY_DISTRIBUTION_CLOSED_RING_MIXTURE_20,
        data_len: int = 2 ** 26,
        spline_region_count: int = 64,
        n_steps: int = 100,
        angle_mode: str = "rand",
        step_size: float = 0.5,
        spline_kind: str = "cubic",
        fill_mode: str = "extrapolate-const",
        resample_every: int = 1,
        oas_steps: int = 10,
        num_threads: int = None,
):
    if num_threads is not None:
        torch.set_num_threads(num_threads)

    dens0 = dp.get_density_by_name(density)
    layers, rots = train(dens0, data_len, n_steps, angle_mode,
                         spline_region_count, step_size, spline_kind,
                         fill_mode, resample_every, oas_steps)
    save_transport(exp_dir, layers, rots)
    val_loss = validate_loss(exp_dir, dens0, layers, rots)
    visualize_densities(exp_dir, dens0, layers, rots)

    return float(val_loss)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_dir", default="results", help="Base directory for storing results")
    # Inspect main function signature to build parser
    for param in inspect.signature(main).parameters.values():
        if param.name == "exp_dir":
            continue
        type = param.annotation
        if param.default is not inspect.Parameter.empty:
            parser.add_argument(f"--{param.name}", default=param.default,
                                type=type, help=f"({type.__name__}) default: {param.default}")
        else:
            parser.add_argument(param.name, type=type, help=f"({type.__name__})")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--config-file", type=Path, help="Path to a YAML file containing the configuration")
    # Prefer command line arguments over config file
    args = parser.parse_args()
    kwargs = {}
    if args.config_file is not None:
        with args.config_file.open() as file:
            config = yaml.safe_load(file)
        kwargs.update(config)
    kwargs.update(vars(args))
    del kwargs["config_file"]
    del kwargs["seed"]

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Create a new directory for the experiment, making sure parallel runs don't overwrite each other
    base_dir = kwargs.pop("base_dir")
    while True:
        exp_dir = Path(base_dir) / str(int(time()))
        if not exp_dir.exists():
            exp_dir.mkdir(parents=True, exist_ok=False)
            break
    with (exp_dir / "config.yaml").open("w") as file:
        yaml.dump(kwargs, file)

    result = main(exp_dir, **kwargs)
    print(f"Final KL divergence: {result}")

    with (exp_dir / "result.txt").open("w") as file:
        file.write(str(result))
