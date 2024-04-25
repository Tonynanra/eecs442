import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm

def train_func(D_pix, D_live, G_pix, G_live, loader, G_optimizer, D_optimizer, G_scaler, D_scaler, numEpochs=20):
  BCE_loss = nn.BCELoss().cuda()
  L1_loss=nn.L1Loss().cuda()
  MSE_loss = nn.MSELoss().cuda()

  H_reals = 0
  H_fakes = 0
  loop = tqdm(loader, leave=True)

  for epoch in range(numEpochs):
    print('Start training epoch %d' % (epoch + 1))
    
    for idx, (pixel, scenery) in enumerate(loop):
        pixel = pixel.to(config.DEVICE)
        scenery = scenery.to(config.DEVICE)

        # Train Discriminators S and P
        with torch.cuda.amp.autocast():
            fake_scenery = G_live(pixel)
            D_live_real = D_live(scenery)
            D_live_fake = D_live(fake_scenery.detach())
            H_reals += D_live_real.mean().item()
            H_fakes += D_live_fake.mean().item()
            D_live_real_loss = MSE_loss(D_live_real, torch.ones_like(D_live_real))
            D_live_fake_loss = MSE_loss(D_live_fake, torch.zeros_like(D_live_fake))
            D_live_loss = D_live_real_loss + D_live_fake_loss

            fake_pixel = G_pix(scenery)
            D_pix_real = D_pix(pixel)
            D_pix_fake = D_pix(fake_pixel.detach())
            D_pix_real_loss = MSE_loss(D_pix_real, torch.ones_like(D_pix_real))
            D_pix_fake_loss = MSE_loss(D_pix_fake, torch.zeros_like(D_pix_fake))
            D_pix_loss = D_pix_real_loss + D_pix_fake_loss

            # put it togethor
            D_loss = (D_live_loss + D_pix_loss) / 2

        D_optimizer.zero_grad()
        D_scaler.scale(D_loss).backward()
        D_scaler.step(D_optimizer)
        D_scaler.update()

        # Train Generators S and P
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_live_fake = D_live(fake_scenery)
            D_pix_fake = D_pix(fake_pixel)
            loss_G_live = MSE_loss(D_live_fake, torch.ones_like(D_live_fake))
            loss_G_pix = MSE_loss(D_pix_fake, torch.ones_like(D_pix_fake))

            # cycle loss
            cycle_pixel = G_pix(fake_scenery)
            cycle_scenery = G_live(fake_pixel)
            cycle_pixel_loss = L1_loss(pixel, cycle_pixel)
            cycle_scenery_loss = L1_loss(scenery, cycle_scenery)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_pixel = G_pix(pixel)
            identity_scenery = G_live(scenery)
            identity_pixel_loss = L1_loss(pixel, identity_pixel)
            identity_scenery_loss = L1_loss(scenery, identity_scenery)

            # add all togethor
            G_loss = (
                loss_G_pix
                + loss_G_live
                + cycle_pixel_loss * config.LAMBDA_CYCLE
                + cycle_scenery_loss * config.LAMBDA_CYCLE
                + identity_scenery_loss * config.LAMBDA_IDENTITY
                + identity_pixel_loss * config.LAMBDA_IDENTITY
            )

        G_optimizer.zero_grad()
        G_scaler.scale(G_loss).backward()
        G_scaler.step(G_optimizer)
        G_scaler.update()


        loop.set_postfix(H_real=H_reals / (idx + 1), H_fake=H_fakes / (idx + 1))


