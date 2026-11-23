from pathlib import Path

import torch
import matplotlib.pyplot as plt

from .config import SEED, LATENT_DIM, NOISE_DIM
from .diffusion import sample_ddpm
from .metrics import generate_samples, transpose_from_class, transpose_from_latent

FIGURE_DIR = Path(".")

def save_figure(path):
    path = FIGURE_DIR / path
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def image_grid(rows, names, columns, title, path):
    plt.figure(figsize=(columns * 1.5, len(rows) * 1.55))
    for row in range(len(rows)):
        for col in range(columns):
            plt.subplot(len(rows), columns, row * columns + col + 1)
            plt.imshow(rows[row][col].reshape(28, 28).detach().cpu(), cmap="gray", vmin=0, vmax=1)
            plt.xticks([])
            plt.yticks([])
            if col == 0:
                plt.ylabel(names[row], fontsize=9)
    plt.suptitle(title)
    save_figure(path)


@torch.no_grad()
def choose_generator_endpoints(name, models, classifier, labels, device):
    noise = torch.randn(512, NOISE_DIM, device=device)
    generated = (models[name](noise) + 1) / 2
    predicted = classifier(generated).argmax(1)
    candidates_a = (predicted == labels[0]).nonzero().flatten()
    candidates_b = (predicted == labels[1]).nonzero().flatten()
    if len(candidates_a) and len(candidates_b):
        a, b = int(candidates_a[0]), int(candidates_b[0])
        method = "first candidates matching the two reference classes, classifier-predicted"
    else:
        a = 0
        alternatives = (predicted != predicted[a]).nonzero().flatten()
        b = int(alternatives[0]) if len(alternatives) else 1
        method = "fallback: first different predicted class" if len(alternatives) else "two classes NOT found; inspect possible collapse or classifier uncertainty"
    return noise[a], noise[b], {
        "selection": method, "predicted_classes": [int(predicted[a]), int(predicted[b])],
        "candidate_pool": 512, "note": "Predictions are an imperfect proxy; inspect endpoints visually."
    }


@torch.no_grad()
def make_figures(models, histories, schedule, test_loader, mean_z, std_z, device):
    for model in models.values():
        model.eval()
    ae, sparse_ae, vae = models["AE"], models["Sparse AE"], models["VAE"]
    classifier, decoder = models["Classifier"], models["Frozen Decoder"]
    images, labels = next(iter(test_loader))
    images, labels = images[:8].to(device), labels[:8].to(device)
    mean, _ = vae.encode(images)
    image_grid(
        [images, ae(images), sparse_ae(images), vae.decode(mean)],
        ["Asli", "Vanilla AE", "Sparse AE", "VAE (mean)"], 8,
        "Figur 1. Perbandingan rekonstruksi", "figure_1_reconstruction.png"
    )

    torch.manual_seed(SEED + 10)
    names = ["VAE", "GAN", "DCGAN", "DDPM"]
    generated = [generate_samples(name, 16, models, schedule, device).cpu() for name in names]
    plt.figure(figsize=(10, 10))
    for panel in range(4):
        for i in range(16):
            row = (panel // 2) * 4 + i // 4
            col = (panel % 2) * 4 + i % 4
            plt.subplot(8, 8, row * 8 + col + 1)
            plt.imshow(generated[panel][i, 0], cmap="gray", vmin=0, vmax=1)
            plt.axis("off")
            if i == 0:
                plt.title(["A. VAE", "B. GAN", "C. DCGAN", "D. DDPM"][panel], fontsize=10)
    plt.suptitle("Figur 2. Random sampling — 16 gambar per model")
    save_figure("figure_2_sampling.png")

    z = classifier.encode(images)
    random_z = mean_z + std_z * torch.randn(8, LATENT_DIM, device=device)
    image_grid(
        [images, transpose_from_class(classifier, labels), transpose_from_latent(classifier, images),
         decoder(z), ae(images), decoder(random_z)],
        ["1. Asli", "2a. Transpose kelas", "2b. Transpose laten", "3a. Decoder beku", "3b. Vanilla AE", "4. Sampling laten"],
        8, "Figur 3. Empat kelompok bukti eksperimen MLP", "figure_3_classifier.png"
    )

    # Pasangan pertama dari kelas 0 dan 9; gunakan pasangan beda kelas lain jika quick subset tidak memuatnya.
    reference = {}
    fallback = {}
    for batch_images, batch_labels in test_loader:
        for image, label in zip(batch_images, batch_labels):
            key = int(label)
            if key not in fallback:
                fallback[key] = image
            if key in [0, 9] and key not in reference:
                reference[key] = image
        if len(reference) == 2:
            break
    if len(reference) < 2:
        keys = list(fallback)[:2]
        if len(keys) < 2:
            raise ValueError("Interpolasi membutuhkan setidaknya dua kelas test.")
        reference = {key: fallback[key] for key in keys}
    classes = list(reference)
    pair = torch.stack(list(reference.values())).to(device)
    alpha = torch.linspace(0, 1, 6, device=device).unsqueeze(1)
    z_ae, z_sparse = ae.encoder(pair.flatten(1)), sparse_ae.encoder(pair.flatten(1))
    z_vae, _ = vae.encode(pair)
    interpolations = [
        ae.decoder((1-alpha)*z_ae[0] + alpha*z_ae[1]),
        sparse_ae.decoder((1-alpha)*z_sparse[0] + alpha*z_sparse[1]),
        vae.decode((1-alpha)*z_vae[0] + alpha*z_vae[1]),
    ]
    interpolation_notes = {"reference_classes": classes, "alpha": [0, .2, .4, .6, .8, 1]}
    torch.manual_seed(SEED + 20)
    for name in ["GAN", "DCGAN"]:
        a, b, notes = choose_generator_endpoints(name, models, classifier, classes, device)
        interpolations.append((models[name]((1-alpha)*a+alpha*b)+1)/2)
        interpolation_notes[name] = notes
    plt.figure(figsize=(11, 9))
    for row in range(5):
        for col in range(6):
            plt.subplot(5, 6, row * 6 + col + 1)
            plt.imshow(interpolations[row][col].reshape(28, 28).cpu(), cmap="gray", vmin=0, vmax=1)
            plt.title(f"alpha={col/5:.1f}")
            plt.xticks([])
            plt.yticks([])
            if col == 0:
                plt.ylabel(["AE", "Sparse AE", "VAE", "GAN", "DCGAN"][row])
    plt.suptitle("Figur 4. Linear latent interpolation")
    save_figure("figure_4_interpolation.png")

    torch.manual_seed(SEED + 30)
    _, trajectory = sample_ddpm(models["DDPM"], 1, schedule, device, return_trajectory=True)
    plt.figure(figsize=(14, 11))
    for i, name in enumerate(["AE", "Sparse AE", "VAE", "GAN", "DCGAN", "DDPM"]):
        plt.subplot(3, 6, (i * 2 + 1, i * 2 + 2))
        history = histories[name]
        epochs = [row["epoch"] for row in history]
        if name in ["GAN", "DCGAN"]:
            plt.plot(epochs, [row["generator_loss"] for row in history], label="Generator", marker=".")
            plt.plot(epochs, [row["discriminator_loss"] for row in history], label="Discriminator", marker=".")
        elif name in ["AE", "Sparse AE"]:
            plt.plot(epochs, [row["train_mse"] for row in history], label="Train reconstruction MSE", marker=".")
            plt.plot(epochs, [row["val_mse"] for row in history], label="Validation MSE", marker=".")
        else:
            plt.plot(epochs, [row["train_loss"] for row in history], label="Train", marker=".")
            plt.plot(epochs, [row["val_loss"] for row in history], label="Validation", marker=".")
            if name == "VAE" and "reconstruction" in history[0] and "kl" in history[0]:
                plt.plot(epochs, [row["reconstruction"] for row in history], label="BCE", marker=".")
                plt.plot(epochs, [row["kl"] for row in history], label="KL", marker=".")
        plt.title(name)
        plt.xlabel("Epoch")
        plt.ylabel("Loss (skala berbeda antar model)")
        plt.legend(fontsize=7)
    for i, (step, image) in enumerate(trajectory):
        plt.subplot(3, 6, 13 + i)
        plt.imshow(image[0, 0], cmap="gray", vmin=0, vmax=1)
        plt.title(f"DDPM x_{step}")
        plt.axis("off")
    plt.suptitle("Figur 5. Loss training dan reverse denoising DDPM")
    save_figure("figure_5_loss_and_diffusion.png")
    return interpolation_notes


def architecture_diagrams():
    structures = [
        ("Vanilla AE", ["x: 784", "Dense 256 / ReLU", "z: 64 / ReLU", "Dense 256 / ReLU", "784 / Sigmoid"]),
        ("Sparse AE", ["x: 784", "Dense 256 / ReLU", "z: 64 + L1", "Dense 256 / ReLU", "784 / Sigmoid"]),
        ("VAE", ["x: 784", "Dense 256 / ReLU", "mean, logvar: 64", "sample z -> 256", "784 / Sigmoid"]),
        ("GAN", ["noise: 100", "G: 256 -> 512", "784 / Tanh", "D: 512 -> 256", "1 logit"]),
        ("DCGAN", ["noise: 100", "G: 7 -> 14 -> 28", "28x28 / Tanh", "D: 14 -> 7 -> 1", "1 logit"]),
        ("DDPM", ["x_t + timestep", "28 -> 14 -> 7", "UNet + time emb", "7 -> 14 -> 28", "noise prediction"]),
        ("MLP Classifier", ["x: 784", "Dense 256 / ReLU", "z: 64 / ReLU", "Dense 10", "10 logits"]),
    ]
    plt.figure(figsize=(15, 11))
    for row, (name, blocks) in enumerate(structures):
        plt.subplot(7, 1, row + 1)
        plt.xlim(0, 6)
        plt.ylim(0, 1)
        plt.axis("off")
        plt.text(0, 0.9, name, fontsize=11, weight="bold")
        for col, block in enumerate(blocks):
            x = 0.6 + col * 1.15
            plt.text(x, 0.45, block, ha="center", va="center", fontsize=9,
                     bbox={"boxstyle": "round,pad=0.4", "facecolor": "#eef3fa", "edgecolor": "#305780"})
            if col < len(blocks)-1:
                plt.annotate("", xy=(x+0.8, .45), xytext=(x+0.43, .45), arrowprops={"arrowstyle": "->"})
    plt.suptitle("Skema arsitektur; GAN discriminator juga menerima gambar asli")
    save_figure("architecture_diagrams.png")


def save_all_figures(output_dir, models, histories, schedule, test_loader, mean_z, std_z, device):
    global FIGURE_DIR
    FIGURE_DIR = Path(output_dir)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    notes = make_figures(models, histories, schedule, test_loader, mean_z, std_z, device)
    architecture_diagrams()
    return notes
