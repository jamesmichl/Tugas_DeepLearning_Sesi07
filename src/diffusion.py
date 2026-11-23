import torch

def diffusion_schedule(steps, device):
    # Skala jadwal hanya berubah pada quick run dengan jumlah langkah kecil.
    beta = torch.linspace(0.0001 * 1000 / steps, min(0.02 * 1000 / steps, 0.999), steps, device=device)
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    alpha_bar_prev = torch.cat([torch.ones(1, device=device), alpha_bar[:-1]])
    variance = beta * (1 - alpha_bar_prev) / (1 - alpha_bar)
    coefficient_x0 = beta * alpha_bar_prev.sqrt() / (1 - alpha_bar)
    coefficient_xt = (1 - alpha_bar_prev) * alpha.sqrt() / (1 - alpha_bar)
    return {
        "beta": beta, "alpha_bar": alpha_bar, "variance": variance,
        "coefficient_x0": coefficient_x0, "coefficient_xt": coefficient_xt,
    }


def add_noise(images, t, noise, schedule):
    alpha_bar = schedule["alpha_bar"][t].reshape(-1, 1, 1, 1)
    return alpha_bar.sqrt() * images + (1 - alpha_bar).sqrt() * noise


@torch.no_grad()
def sample_ddpm(model, count, schedule, device, return_trajectory=False):
    model.eval()
    steps = len(schedule["beta"])
    x = torch.randn(count, 1, 28, 28, device=device)
    milestones = sorted(set([steps, int(steps * 0.8), int(steps * 0.6), int(steps * 0.4), int(steps * 0.2), 0]), reverse=True)
    trajectory = [(steps, ((x[:1] + 1) / 2).clamp(0, 1).cpu())] if return_trajectory else []
    for step in range(steps - 1, -1, -1):
        t = torch.full((count,), step, dtype=torch.long, device=device)
        predicted_noise = model(x, t)
        alpha_bar = schedule["alpha_bar"][step]
        predicted_x0 = ((x - (1 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt()).clamp(-1, 1)
        mean = schedule["coefficient_x0"][step] * predicted_x0 + schedule["coefficient_xt"][step] * x
        if step > 0:
            x = mean + schedule["variance"][step].sqrt() * torch.randn_like(x)
        else:
            x = mean
        if return_trajectory and step in milestones:
            trajectory.append((step, ((x[:1] + 1) / 2).clamp(0, 1).cpu()))
    samples = ((x + 1) / 2).clamp(0, 1)
    return (samples, trajectory) if return_trajectory else samples
