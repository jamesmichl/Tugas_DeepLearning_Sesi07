import time

import torch
import torch.nn as nn
import torch.optim as optim

from .config import SEED, MAX_EPOCHS, LEARNING_RATE, NOISE_DIM, SPARSE_LAMBDA
from .models import Autoencoder, VAE
from .diffusion import add_noise
from .utils import sync_gpu

bce_loss = nn.BCEWithLogitsLoss(reduction="sum")

def vae_loss(logits, images, mean, log_var):
    reconstruction = bce_loss(logits, images.flatten(1)) / images.size(0)
    kl = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / images.size(0)
    return reconstruction + kl


def train_ae(train_loader, val_loader, device):
    torch.manual_seed(SEED)
    train_loader.generator.manual_seed(SEED)

    ae = Autoencoder().to(device)
    ae_optimizer = optim.Adam(ae.parameters(), lr=LEARNING_RATE)
    ae_loss = nn.MSELoss()

    ae_train_loss = []
    ae_val_loss = []
    ae_epoch_time = []

    for epoch in range(MAX_EPOCHS):
        # Training
        ae.train()
        total_loss = 0

        sync_gpu(device)
        start = time.perf_counter()

        for images, _ in train_loader:
            images = images.to(device)

            ae_optimizer.zero_grad()
            outputs = ae(images)
            loss = ae_loss(outputs, images.flatten(1))
            loss.backward()
            ae_optimizer.step()

            total_loss += loss.item() * images.size(0)

        sync_gpu(device)
        ae_epoch_time.append(time.perf_counter() - start)
        ae_train_loss.append(total_loss / len(train_loader.dataset))

        # Validation
        ae.eval()
        total_val = 0

        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                outputs = ae(images)
                loss = ae_loss(outputs, images.flatten(1))

                total_val += loss.item() * images.size(0)

        ae_val_loss.append(total_val / len(val_loader.dataset))

        print(
            f"Epoch {epoch+1}: "
            f"train={ae_train_loss[-1]:.5f}, "
            f"val={ae_val_loss[-1]:.5f}"
        )

    history = []
    for i in range(len(ae_train_loss)):
        history.append({"epoch": i + 1, "train_mse": ae_train_loss[i],
                        "val_mse": ae_val_loss[i], "seconds": ae_epoch_time[i]})
    ae.eval()
    return ae, history



def train_vae(train_loader, val_loader, device):
    torch.manual_seed(SEED)
    train_loader.generator.manual_seed(SEED)
    vae = VAE().to(device)
    vae_optimizer = optim.Adam(vae.parameters(), lr=LEARNING_RATE)

    vae_train_loss = []
    vae_val_loss = []
    vae_epoch_time = []
    best_val = float("inf")

    for epoch in range(MAX_EPOCHS):
        vae.train()
        total_loss = 0
        sync_gpu(device)
        start = time.perf_counter()

        for images, _ in train_loader:
            images = images.to(device)
            vae_optimizer.zero_grad()
            logits, mean, log_var = vae(images)
            loss = vae_loss(logits, images, mean, log_var)
            loss.backward()
            vae_optimizer.step()
            total_loss += loss.item() * images.size(0)

        sync_gpu(device)
        vae_epoch_time.append(time.perf_counter() - start)
        vae_train_loss.append(total_loss / len(train_loader.dataset))

        vae.eval()
        total_val = 0
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                logits, mean, log_var = vae(images)
                loss = vae_loss(logits, images, mean, log_var)
                total_val += loss.item() * images.size(0)
        val_loss = total_val / len(val_loader.dataset)
        vae_val_loss.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            vae_best_epoch = epoch + 1
            vae_best_weights = {k: v.detach().cpu().clone() for k, v in vae.state_dict().items()}

        print(f"Epoch {epoch+1}: train={vae_train_loss[-1]:.3f}, val={val_loss:.3f}")

    vae.load_state_dict(vae_best_weights)
    print("Checkpoint VAE: epoch", vae_best_epoch)

    history = []
    for i in range(len(vae_train_loss)):
        history.append({"epoch": i + 1, "train_loss": vae_train_loss[i],
                        "val_loss": vae_val_loss[i], "seconds": vae_epoch_time[i]})
    vae.eval()
    return vae, history



def train_sparse_ae(train_loader, val_loader, device):
    torch.manual_seed(SEED)
    train_loader.generator.manual_seed(SEED)

    sparse_ae = Autoencoder().to(device)
    sparse_optimizer = optim.Adam(sparse_ae.parameters(), lr=LEARNING_RATE)
    sparse_loss = nn.MSELoss()

    sparse_train_loss = []
    sparse_val_loss = []
    sparse_epoch_time = []

    for epoch in range(MAX_EPOCHS):
        sparse_ae.train()
        total_loss = 0
        sync_gpu(device)
        start = time.perf_counter()

        for images, _ in train_loader:
            images = images.to(device)
            sparse_optimizer.zero_grad()

            z = sparse_ae.encoder(images.flatten(1))
            outputs = sparse_ae.decoder(z)
            reconstruction = sparse_loss(outputs, images.flatten(1))
            penalty = z.abs().sum(dim=1).mean()
            loss = reconstruction + SPARSE_LAMBDA * penalty

            loss.backward()
            sparse_optimizer.step()
            total_loss += reconstruction.item() * images.size(0)

        sync_gpu(device)
        sparse_epoch_time.append(time.perf_counter() - start)
        sparse_train_loss.append(total_loss / len(train_loader.dataset))

        sparse_ae.eval()
        total_val = 0
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                loss = sparse_loss(sparse_ae(images), images.flatten(1))
                total_val += loss.item() * images.size(0)

        sparse_val_loss.append(total_val / len(val_loader.dataset))
        print(f"Epoch {epoch+1}: train MSE={sparse_train_loss[-1]:.5f}, val MSE={sparse_val_loss[-1]:.5f}")

    history = []
    for i in range(len(sparse_train_loss)):
        history.append({"epoch": i + 1, "train_mse": sparse_train_loss[i],
                        "val_mse": sparse_val_loss[i], "seconds": sparse_epoch_time[i]})
    sparse_ae.eval()
    return sparse_ae, history



def train_gan(generator, discriminator, g_optimizer, d_optimizer, train_loader, device, name):
    criterion = nn.BCEWithLogitsLoss()
    history = []
    for epoch in range(MAX_EPOCHS):
        generator.train()
        g_total, d_total, count = 0.0, 0.0, 0
        sync_gpu(device)
        start = time.perf_counter()
        for images, _ in train_loader:
            real_images = images.to(device) * 2 - 1
            n = real_images.size(0)
            discriminator.train()
            for parameter in discriminator.parameters():
                parameter.requires_grad_(True)
            noise = torch.randn(n, NOISE_DIM, device=device)
            fake_images = generator(noise)
            d_optimizer.zero_grad()
            real_logits = discriminator(real_images)
            fake_logits = discriminator(fake_images.detach())
            d_loss = (criterion(real_logits, torch.ones_like(real_logits)) +
                      criterion(fake_logits, torch.zeros_like(fake_logits))) / 2
            d_loss.backward()
            d_optimizer.step()

            # Bobot dan statistik BatchNorm discriminator tidak berubah saat update G.
            discriminator.eval()
            for parameter in discriminator.parameters():
                parameter.requires_grad_(False)
            g_optimizer.zero_grad()
            fake_logits = discriminator(fake_images)
            g_loss = criterion(fake_logits, torch.ones_like(fake_logits))
            g_loss.backward()
            g_optimizer.step()
            g_total += g_loss.item() * n
            d_total += d_loss.item() * n
            count += n
        sync_gpu(device)
        row = {"epoch": epoch+1, "generator_loss": g_total/count,
               "discriminator_loss": d_total/count, "seconds": time.perf_counter()-start}
        history.append(row)
        print(f"{name} {epoch+1}: G={row['generator_loss']:.3f}, D={row['discriminator_loss']:.3f}")
    for parameter in discriminator.parameters():
        parameter.requires_grad_(True)
    generator.eval()
    discriminator.eval()
    return history


def train_ddpm(model, optimizer, train_loader, val_loader, schedule, device):
    criterion = nn.MSELoss()
    history = []
    steps = len(schedule["beta"])
    for epoch in range(MAX_EPOCHS):
        model.train()
        total, count = 0.0, 0
        sync_gpu(device)
        start = time.perf_counter()
        for images, _ in train_loader:
            images = images.to(device) * 2 - 1
            t = torch.randint(0, steps, (images.size(0),), device=device)
            noise = torch.randn_like(images)
            noisy_images = add_noise(images, t, noise, schedule)
            optimizer.zero_grad()
            loss = criterion(model(noisy_images, t), noise)
            loss.backward()
            optimizer.step()
            total += loss.item() * images.size(0)
            count += images.size(0)
        sync_gpu(device)
        seconds = time.perf_counter() - start
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device) * 2 - 1
                t = torch.randint(0, steps, (images.size(0),), device=device)
                noise = torch.randn_like(images)
                loss = criterion(model(add_noise(images, t, noise, schedule), t), noise)
                val_total += loss.item() * images.size(0)
        row = {"epoch": epoch+1, "train_loss": total/count,
               "val_loss": val_total/len(val_loader.dataset), "seconds": seconds}
        history.append(row)
        print(f"DDPM {epoch+1}: noise MSE={row['train_loss']:.5f}, val={row['val_loss']:.5f}")
    model.eval()
    return history


def train_classifier(model, optimizer, train_loader, val_loader, device):
    criterion = nn.CrossEntropyLoss()
    history = []
    for epoch in range(MAX_EPOCHS):
        model.train()
        total, correct, count = 0.0, 0, 0
        sync_gpu(device)
        start = time.perf_counter()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            count += images.size(0)
        sync_gpu(device)
        seconds = time.perf_counter()-start
        model.eval()
        val_correct, val_loss = 0, 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_loss += criterion(logits, labels).item() * images.size(0)
        row = {"epoch": epoch+1, "train_loss": total/count, "train_accuracy": correct/count,
               "val_loss": val_loss/len(val_loader.dataset),
               "val_accuracy": val_correct/len(val_loader.dataset), "seconds": seconds}
        history.append(row)
        print(f"MLP {epoch+1}: train acc={row['train_accuracy']:.3f}, val acc={row['val_accuracy']:.3f}")
    model.eval()
    return history


def train_frozen_decoder(classifier, decoder, optimizer, train_loader, val_loader, device):
    classifier.eval()
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)
    criterion = nn.MSELoss()
    history = []
    for epoch in range(MAX_EPOCHS):
        decoder.train()
        total, count = 0.0, 0
        sync_gpu(device)
        start = time.perf_counter()
        for images, _ in train_loader:
            images = images.to(device)
            with torch.no_grad():
                features = classifier.encode(images)
            optimizer.zero_grad()
            loss = criterion(decoder(features), images.flatten(1))
            loss.backward()
            optimizer.step()
            total += loss.item() * images.size(0)
            count += images.size(0)
        sync_gpu(device)
        seconds = time.perf_counter()-start
        decoder.eval()
        val_total = 0.0
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                val_total += criterion(decoder(classifier.encode(images)), images.flatten(1)).item()*images.size(0)
        row = {"epoch": epoch+1, "train_loss": total/count,
               "val_loss": val_total/len(val_loader.dataset), "seconds": seconds}
        history.append(row)
        print(f"Frozen decoder {epoch+1}: MSE={row['train_loss']:.5f}, val={row['val_loss']:.5f}")
    decoder.eval()
    return history
