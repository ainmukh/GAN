import torch
from src.utils import adv_loss, r1_reg
import wandb


def compute_g_loss(nets, args, x_real, y_org, y_trg, step: int, z_trgs=None, x_refs=None):
    assert (z_trgs is None) != (x_refs is None)
    if z_trgs is not None:
        z_trg, z_trg2 = z_trgs
    if x_refs is not None:
        x_ref, x_ref2 = x_refs

    # adversarial loss
    if z_trgs is not None:
        s_trg = nets['mapping'](z_trg, y_trg)
    else:
        s_trg = nets['style'](x_ref, y_trg)

    x_fake = nets['generator'](x_real, s_trg)
    out = nets['discriminator'](x_fake, y_trg)
    loss_adv = adv_loss(out, 1)

    # style reconstruction loss
    s_pred = nets['style'](x_fake, y_trg)
    loss_sty = torch.mean(torch.abs(s_pred - s_trg))

    # diversity sensitive loss
    if z_trgs is not None:
        s_trg2 = nets['mapping'](z_trg2, y_trg)
    else:
        s_trg2 = nets['style'](x_ref2, y_trg)
    x_fake2 = nets['generator'](x_real, s_trg2)
    x_fake2 = x_fake2.detach()
    loss_ds = torch.mean(torch.abs(x_fake - x_fake2))

    # cycle-consistency loss
    s_org = nets['style'](x_real, y_org)
    x_rec = nets['generator'](x_fake, s_org)
    loss_cyc = torch.mean(torch.abs(x_rec - x_real))

    loss = loss_adv + args.lambda_sty * loss_sty \
        - args.lambda_ds * loss_ds + args.lambda_cyc * loss_cyc

    if step % 10 == 0:
        images = wandb.Image(torch.cat([x_real, x_fake, x_fake2]), caption='Real; Fake; Fake')
        wandb.log({'Images': images})

    return loss, {'adv': loss_adv.item(), 'sty': loss_sty.item(), 'ds': loss_ds.item(), 'cyc': loss_cyc.item()}


def compute_d_loss(nets, args, x_real, y_org, y_trg, z_trg=None, x_ref=None):
    assert (z_trg is None) != (x_ref is None)
    # with real images
    x_real.requires_grad_()
    out = nets['discriminator'](x_real, y_org)
    loss_real = adv_loss(out, 1)
    loss_reg = r1_reg(out, x_real)

    # with fake images
    with torch.no_grad():
        if z_trg is not None:
            s_trg = nets.mapping_network(z_trg, y_trg)
        else:  # x_ref is not None
            s_trg = nets.style_encoder(x_ref, y_trg)

        x_fake = nets['generator'](x_real, s_trg)
    out = nets['discriminator'](x_fake, y_trg)
    loss_fake = adv_loss(out, 0)

    loss = loss_real + loss_fake + args.lambda_reg * loss_reg
    return loss, {'real': loss_real.item(), 'fake': loss_fake.item(), 'reg': loss_reg.item()}
