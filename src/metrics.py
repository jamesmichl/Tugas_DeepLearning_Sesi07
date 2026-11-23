import time

import torch
from skimage.metrics import structural_similarity

from .config import SEED, LATENT_DIM, NOISE_DIM, FID_SAMPLES, FID_BATCH_SIZE
from .diffusion import sample_ddpm
from .utils import sync_gpu

@torch.no_grad()
def reconstruction_metrics(operation, test_loader, device):
    mse_total, psnr_total, ssim_total, count = 0.0, 0.0, 0.0, 0
    for images, _ in test_loader:
        predicted = operation(images.to(device)).reshape(-1, 1, 28, 28).cpu()
        mse = ((images - predicted)**2).flatten(1).mean(1)
        mse_total += mse.sum().item()
        psnr_total += (-10 * torch.log10(mse.clamp_min(1e-12))).sum().item()
        for original, reconstruction in zip(images, predicted):
            ssim_total += structural_similarity(
                original[0].numpy(), reconstruction[0].numpy(), data_range=1.0, win_size=7
            )
        count += images.size(0)
    return {"MSE": mse_total/count, "PSNR": psnr_total/count, "SSIM": float(ssim_total/count)}


@torch.no_grad()
def classifier_accuracy(classifier, test_loader, device):
    classifier.eval()
    correct, count = 0, 0
    for images, labels in test_loader:
        logits = classifier(images.to(device))
        correct += (logits.argmax(1).cpu() == labels).sum().item()
        count += images.size(0)
    return correct / count


@torch.no_grad()
def latent_statistics(classifier, test_loader, device):
    classifier.eval()
    features = []
    for images, _ in test_loader:
        features.append(classifier.encode(images.to(device)).cpu())
    features = torch.cat(features)
    return features.mean(0).to(device), features.std(0, unbiased=False).to(device)


@torch.no_grad()
def latent_sparsity(model, test_loader, device):
    model.eval()
    zero_count, absolute_sum, count = 0, 0.0, 0
    for images, _ in test_loader:
        z = model.encoder(images.to(device).flatten(1))
        zero_count += (z.abs() < 1e-6).sum().item()
        absolute_sum += z.abs().sum().item()
        count += z.numel()
    return {"zero_fraction": zero_count/count, "mean_absolute_activation": absolute_sum/count}


@torch.no_grad()
def transpose_from_class(classifier, labels):
    # Baris data dikalikan W, ekuivalen dengan W^T pada notasi vektor kolom di PDF.
    one_hot = torch.eye(10, device=labels.device)[labels]
    z = torch.relu(one_hot @ classifier.fc3.weight)
    hidden = torch.relu(z @ classifier.fc2.weight)
    return torch.sigmoid(hidden @ classifier.fc1.weight)


@torch.no_grad()
def transpose_from_latent(classifier, images):
    z = classifier.encode(images)
    hidden = torch.relu(z @ classifier.fc2.weight)
    return torch.sigmoid(hidden @ classifier.fc1.weight)


@torch.no_grad()
def weight_diagnostics(classifier):
    result = {}
    for name, layer in [("W1", classifier.fc1), ("W2", classifier.fc2), ("W3", classifier.fc3)]:
        weight = layer.weight.detach().cpu()
        identity = torch.eye(weight.size(1))
        result[name] = {
            "shape": list(weight.shape), "rank": int(torch.linalg.matrix_rank(weight)),
            "relative_WtW_minus_I": float(torch.linalg.matrix_norm(weight.T @ weight-identity) /
                                         torch.linalg.matrix_norm(identity))
        }
    return result


@torch.no_grad()
def generate_samples(name, count, models, schedule, device):
    if name == "DDPM":
        return sample_ddpm(models[name], count, schedule, device)
    dimension = LATENT_DIM if name == "VAE" else NOISE_DIM
    noise = torch.randn(count, dimension, device=device)
    if name == "VAE":
        return models[name].decode(noise).reshape(-1, 1, 28, 28)
    return (models[name](noise) + 1) / 2


@torch.no_grad()
def calculate_fid(name, models, schedule, test_loader, device):
    from torchmetrics.image.fid import FrechetInceptionDistance
    # 2048-dimensi Inception; channel grayscale diulang menjadi RGB.
    # Resize 299x299 ditangani extractor torch-fidelity untuk real dan fake.
    metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    metric.set_dtype(torch.float64)
    n = min(FID_SAMPLES, len(test_loader.dataset))
    processed = 0
    for images, _ in test_loader:
        real = images[:n-processed].to(device)
        metric.update(real.repeat(1, 3, 1, 1), real=True)
        processed += real.size(0)
        if processed >= n:
            break
    processed = 0
    while processed < n:
        count = min(FID_BATCH_SIZE, n-processed)
        fake = generate_samples(name, count, models, schedule, device).clamp(0, 1)
        metric.update(fake.repeat(1, 3, 1, 1), real=False)
        processed += count
        print(f"FID {name}: {processed}/{n}", end="\r")
    score = float(metric.compute())
    print(f"FID {name}: {score:.4f} ({n} real, {n} generated)")
    return score


@torch.no_grad()
def measure_latency(operation, device, warmup, repeats):
    for _ in range(warmup):
        operation()
    sync_gpu(device)
    start = time.perf_counter()
    for _ in range(repeats):
        operation()
    sync_gpu(device)
    return (time.perf_counter()-start) * 1000 / repeats


def evaluate_all(models, histories, schedule, test_loader, device, run_fid=True, progress_path=None):
    for model in models.values():
        model.eval()
    ae, sparse_ae, vae = models["AE"], models["Sparse AE"], models["VAE"]
    classifier, frozen_decoder = models["Classifier"], models["Frozen Decoder"]
    operations = {
        "AE": lambda x: ae(x),
        "Sparse AE": lambda x: sparse_ae(x),
        "VAE": lambda x: vae.decode(vae.encode(x)[0]),
        "Frozen Decoder": lambda x: frozen_decoder(classifier.encode(x)),
        "Latent Transpose": lambda x: transpose_from_latent(classifier, x),
    }
    reconstruction = {name: reconstruction_metrics(operation, test_loader, device)
                      for name, operation in operations.items()}
    accuracy = classifier_accuracy(classifier, test_loader, device)
    print(f"MLP test accuracy: {accuracy:.2%}; target >= 88%: {accuracy >= 0.88}")
    if accuracy < 0.88:
        print("Target classifier BELUM terpenuhi. Perbaiki konfigurasi berdasarkan validation; jangan mengklaim sudah lolos.")
    mean_z, std_z = latent_statistics(classifier, test_loader, device)
    extra = {
        "classifier_test_accuracy": accuracy, "classifier_target_met": accuracy >= 0.88,
        "ae_sparsity": latent_sparsity(ae, test_loader, device),
        "sparse_ae_sparsity": latent_sparsity(sparse_ae, test_loader, device),
        "weights": weight_diagnostics(classifier),
        "frozen_decoder": reconstruction["Frozen Decoder"],
        "latent_transpose": reconstruction["Latent Transpose"],
        "test_latent_prior_note": "Diagnostic only: mean/std use test features as explicitly requested; not an independent generative benchmark.",
    }
    batch = next(iter(test_loader))[0][:64].to(device)
    vae_noise = torch.randn(64, LATENT_DIM, device=device)
    gan_noise = torch.randn(64, NOISE_DIM, device=device)
    latency_operations = {
        "AE": lambda: ae(batch), "Sparse AE": lambda: sparse_ae(batch),
        "VAE": lambda: vae.decode(vae_noise),
        "GAN": lambda: (models["GAN"](gan_noise)+1)/2,
        "DCGAN": lambda: (models["DCGAN"](gan_noise)+1)/2,
        "DDPM": lambda: sample_ddpm(models["DDPM"], 64, schedule, device),
    }
    rows = []
    def save_progress():
        if progress_path is not None:
            torch.save({"metrics": rows, "classifier_results": extra,
                        "classifier_latent_mean": mean_z.cpu(),
                        "classifier_latent_std": std_z.cpu(), "histories": histories}, progress_path)
    save_progress()
    for name in ["AE", "Sparse AE", "VAE", "GAN", "DCGAN", "DDPM"]:
        row = {
            "model": name,
            "latent_dimension": "1x28x28 noise" if name == "DDPM" else
                                NOISE_DIM if name in ["GAN", "DCGAN"] else LATENT_DIM,
            "training_seconds_per_epoch": sum(h["seconds"] for h in histories[name])/len(histories[name]),
            "latency_ms_batch64": measure_latency(latency_operations[name], device,
                                                  2, 5),
            "latency_operation": "reconstruction" if name in ["AE", "Sparse AE"] else "generation",
            "MSE": reconstruction.get(name, {}).get("MSE"),
            "PSNR": reconstruction.get(name, {}).get("PSNR"),
            "SSIM": reconstruction.get(name, {}).get("SSIM"),
            "FID": None,
            "stability_notes": "Isi berdasarkan kurva dan sampel aktual",
            "quick_run": False,
        }
        rows.append(row)
        save_progress()
        if run_fid and name in ["VAE", "GAN", "DCGAN", "DDPM"]:
            torch.manual_seed(SEED + 123)
            row["FID"] = calculate_fid(name, models, schedule, test_loader, device)
        save_progress()
        print(name, row)
    return rows, extra, mean_z, std_z
