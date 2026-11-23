from pathlib import Path
import platform

import torch
import torch.optim as optim
import torchvision

from . import config
from .config import SEED, LEARNING_RATE, LATENT_DIM, HIDDEN_DIM, NOISE_DIM, DIFFUSION_STEPS
from .data import load_data
from .models import (Autoencoder, VAE, Generator, Discriminator, DCGenerator,
                     DCDiscriminator, initialize_dcgan, Denoiser, MLPClassifier, FrozenDecoder)
from .diffusion import diffusion_schedule
from .training import (train_ae, train_sparse_ae, train_vae, train_gan,
                       train_ddpm, train_classifier, train_frozen_decoder)
from .metrics import evaluate_all
from .figures import save_all_figures
from .utils import set_seed, save_json, save_csv

WEIGHT_FILES = {
    "AE": "ae_weights.pt", "Sparse AE": "sparse_ae_weights.pt", "VAE": "vae_weights.pt",
    "GAN": "gan_weights.pt", "DCGAN": "dcgan_weights.pt", "DDPM": "ddpm_weights.pt",
    "Classifier": "classifier_weights.pt", "Frozen Decoder": "frozen_decoder_weights.pt"
}


def reset_training_seed(train_loader):
    torch.manual_seed(SEED)
    train_loader.generator.manual_seed(SEED)


def train_all(train_loader, val_loader, device, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models, histories = {}, {}

    def save_model(name, model, history):
        models[name], histories[name] = model, history
        torch.save(model.state_dict(), output_dir / WEIGHT_FILES[name])
        torch.save(histories, output_dir / "histories.pt")
        save_csv(output_dir / (WEIGHT_FILES[name].replace("_weights.pt", "_history.csv")), history)

    for name, train_function in [("AE", train_ae), ("VAE", train_vae), ("Sparse AE", train_sparse_ae)]:
        print("Training", name)
        model, history = train_function(train_loader, val_loader, device)
        save_model(name, model, history)

    for name in ["GAN", "DCGAN"]:
        reset_training_seed(train_loader)
        if name == "GAN":
            generator, discriminator = Generator(NOISE_DIM).to(device), Discriminator().to(device)
        else:
            generator, discriminator = DCGenerator(NOISE_DIM).to(device), DCDiscriminator().to(device)
            generator.apply(initialize_dcgan)
            discriminator.apply(initialize_dcgan)
        g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        history = train_gan(generator, discriminator, g_optimizer, d_optimizer, train_loader, device, name)
        save_model(name, generator, history)
        torch.save(discriminator.state_dict(), output_dir / f"{name.lower()}_discriminator_weights.pt")

    reset_training_seed(train_loader)
    ddpm = Denoiser().to(device)
    schedule = diffusion_schedule(DIFFUSION_STEPS, device)
    optimizer = optim.Adam(ddpm.parameters(), lr=0.0002)
    save_model("DDPM", ddpm, train_ddpm(ddpm, optimizer, train_loader, val_loader, schedule, device))

    reset_training_seed(train_loader)
    classifier = MLPClassifier(LATENT_DIM, HIDDEN_DIM).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=LEARNING_RATE)
    save_model("Classifier", classifier, train_classifier(classifier, optimizer, train_loader, val_loader, device))
    torch.save({"W1": classifier.fc1.weight.detach().cpu(),
                "W2": classifier.fc2.weight.detach().cpu(),
                "W3": classifier.fc3.weight.detach().cpu()}, output_dir / "classifier_matrices.pt")

    reset_training_seed(train_loader)
    decoder = FrozenDecoder(LATENT_DIM, HIDDEN_DIM).to(device)
    optimizer = optim.Adam(decoder.parameters(), lr=LEARNING_RATE)
    save_model("Frozen Decoder", decoder,
               train_frozen_decoder(classifier, decoder, optimizer, train_loader, val_loader, device))
    return models, histories, schedule


def load_models(output_dir, device):
    output_dir = Path(output_dir)
    constructors = {"AE": Autoencoder, "Sparse AE": Autoencoder, "VAE": VAE,
                    "GAN": Generator, "DCGAN": DCGenerator, "DDPM": Denoiser,
                    "Classifier": MLPClassifier, "Frozen Decoder": FrozenDecoder}
    models = {}
    for name, constructor in constructors.items():
        model = constructor().to(device)
        model.load_state_dict(torch.load(output_dir / WEIGHT_FILES[name], map_location="cpu", weights_only=True))
        models[name] = model.eval()
    histories = torch.load(output_dir / "histories.pt", map_location="cpu", weights_only=True)
    return models, histories, diffusion_schedule(DIFFUSION_STEPS, device)


def run_experiment(output_dir, data_dir="data", evaluate_only=False, figures_only=False, run_fid=True):
    set_seed()
    output_dir = Path(output_dir)
    if not evaluate_only and not figures_only and output_dir.exists() and any(output_dir.glob("*_weights.pt")):
        raise FileExistsError("Folder sudah memiliki bobot. Gunakan --evaluate-only atau pilih --output-dir baru.")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    train_loader, val_loader, test_loader = load_data(data_dir, output_dir / "split_indices.pt")

    if evaluate_only or figures_only:
        models, histories, schedule = load_models(output_dir, device)
    else:
        settings = {name: value for name, value in vars(config).items() if name.isupper()}
        settings.update({"python": platform.python_version(), "torch": torch.__version__,
                         "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
                         "device": str(device),
                         "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                         "split_sizes": [56000, 7000, 7000]})
        save_json(output_dir / "run_config.json", settings)
        models, histories, schedule = train_all(train_loader, val_loader, device, output_dir)

    if figures_only:
        result = torch.load(output_dir / "evaluation_results.pt", map_location="cpu", weights_only=True)
        mean_z = result["classifier_latent_mean"].to(device)
        std_z = result["classifier_latent_std"].to(device)
    else:
        rows, extra, mean_z, std_z = evaluate_all(
            models, histories, schedule, test_loader, device, run_fid=run_fid,
            progress_path=output_dir / "evaluation_partial.pt"
        )
        result = {"metrics": rows, "classifier_results": extra,
                  "classifier_latent_mean": mean_z.cpu(), "classifier_latent_std": std_z.cpu(),
                  "histories": histories}
        # Simpan sebelum plotting, agar hasil evaluasi tidak hilang jika gambar gagal.
        torch.save(result, output_dir / "evaluation_results.pt")
        save_csv(output_dir / "metrics.csv", rows)
        save_json(output_dir / "classifier_results.json", extra)

    notes = save_all_figures(output_dir / "figures", models, histories, schedule,
                            test_loader, mean_z, std_z, device)
    result["interpolation_notes"] = notes
    torch.save(result, output_dir / "evaluation_results.pt")
    save_json(output_dir / "interpolation_notes.json", notes)
    print("Hasil tersimpan di", output_dir)
