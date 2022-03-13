import torch
from src.losses import compute_g_loss, compute_d_loss
from tqdm.auto import tqdm
from dataclasses import dataclass
import wandb


def train_epoch(dataloader, ref_loader, nets: dict, optims: dict, args: dataclass, device):
    step = 0
    for inputs in tqdm(zip(dataloader, ref_loader)):
        # fetch images and labels
        # WHAT ARE THESE
        real, ref = inputs
        x_real, y_org = real
        x_ref, x_ref2, y_trg = ref
        x_real, y_org = x_real.to(device), y_org.to(device)
        x_ref, x_ref2, y_trg = x_ref.to(device), x_ref2.to(device), y_trg.to(device)
        z_trg = torch.randn(x_real.size(0), args.latent_dim)
        z_trg2 = torch.randn(x_real.size(0), args.latent_dim)

        # train the discriminator
        d_loss, d_losses_latent = compute_d_loss(nets, args, x_real, y_org, y_trg, z_trg=z_trg)
        for optimizer in optims.values():
            optimizer.zero_grad()
        d_loss.backward()
        optims['discriminator'].step()

        d_loss, d_losses_ref = compute_d_loss(nets, args, x_real, y_org, y_trg, x_ref=x_ref)
        for optimizer in optims.values():
            optimizer.zero_grad()
        d_loss.backward()
        optims['discriminator'].step()

        # train the generator
        g_loss, g_losses_latent = compute_g_loss(nets, args, x_real, y_org, y_trg, step, z_trgs=[z_trg, z_trg2])
        for optimizer in optims.values():
            optimizer.zero_grad()
        g_loss.backward()
        optims['generator'].step()
        optims['mapping'].step()
        optims['style'].step()

        g_loss, g_losses_ref = compute_g_loss(nets, args, x_real, y_org, y_trg, step, x_refs=[x_ref, x_ref2])
        for optimizer in optims.values():
            optimizer.zero_grad()
        g_loss.backward()
        optims['generator'].step()

        if step % 5 == 0:
            wandb.log({'D loss': d_loss, 'G loss': g_loss})
        step += 1


def train(dataloader, ref_loader, nets: dict, optims: dict, args: dataclass, device, num_epochs: int = 10):

    wandb.init(project='GAN_HW2_V2')

    for i in range(num_epochs):
        train_epoch(dataloader, ref_loader, nets, optims, args, device)
