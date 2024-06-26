import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from tqdm import tqdm
from Discriminator_442 import discriminator
from generator import generator, gen_with_attn, attnConfig
from dataset import PixelSceneryDataset
import multiprocessing as mp
import pickle
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from utils import get_checkpoint
import os
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary


#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

def train_func(G_live, G_pix, D_live, D_pix, train_loader, val_loader, G_optimizer, D_optimizer, G_scaler, D_scaler, start_epoch=0, numEpochs=100):
	'''
	Using LSGAN loss for both discriminators and generators, 
	and L1 loss for cycle and identity loss.
	'''
	
	L1_loss=nn.L1Loss().to(device)
	MSE_loss = nn.MSELoss().to(device)

	H_reals = 0
	H_fakes = 0
	writer = SummaryWriter()
	checkpoint_interval = 25
	os.makedirs('checkpoints', exist_ok=True)

	for epoch in range(start_epoch, numEpochs):
		print('Start training epoch %d' % (epoch + 1))

		G_live.train()
		G_pix.train()
		D_live.train()
		D_pix.train()
		train_loop = tqdm(train_loader)
		for idx, (pixel, scenery) in enumerate(train_loop):
			pixel = pixel.to(device, non_blocking=True)
			scenery = scenery.to(device, non_blocking=True)

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
						loss_G_pix * LAMBDA_GEN
						+ loss_G_live * LAMBDA_GEN
						+ cycle_pixel_loss * LAMBDA_CYCLE
						+ cycle_scenery_loss * LAMBDA_CYCLE
						+ identity_scenery_loss * LAMBDA_IDENTITY
						+ identity_pixel_loss * LAMBDA_IDENTITY
				)

			G_optimizer.zero_grad()
			G_scaler.scale(G_loss).backward()
			G_scaler.step(G_optimizer)
			G_scaler.update()
			
			if idx % (len(train_dataset) / BATCH_SIZE // 4) == 0:
				save_folder = f"saved_images/{epoch}"
				os.makedirs(save_folder, exist_ok=True)
				save_image(fake_scenery * 0.5 + 0.5, os.path.join(save_folder, f"scenery_{idx}.png"))
				save_image(fake_pixel * 0.5 + 0.5, os.path.join(save_folder, f"pixel_{idx}.png"))
				writer.add_scalar("D_loss", D_loss.item(), epoch * len(train_loader) + idx)
				writer.add_scalar("D_live_loss", D_live_loss.item(), epoch * len(train_loader) + idx)
				writer.add_scalar("D_pix_loss", D_pix_loss.item(), epoch * len(train_loader) + idx)
				writer.add_scalar("G_loss", G_loss.item(), epoch * len(train_loader) + idx)
				writer.add_scalar("G_live_loss", loss_G_live.item(), epoch * len(train_loader) + idx)
				writer.add_scalar("G_pix_loss", loss_G_pix.item(), epoch * len(train_loader) + idx)
				writer.add_scalar("cycle_pixel_loss", cycle_pixel_loss.item(), epoch * len(train_loader) + idx)
				writer.add_scalar("cycle_scenery_loss", cycle_scenery_loss.item(), epoch * len(train_loader) + idx)
				# writer.add_scalar("identity_pixel_loss", identity_pixel_loss.item(), epoch * len(train_loader) + idx)
				# writer.add_scalar("identity_scenery_loss", identity_scenery_loss.item(), epoch * len(train_loader) + idx)
				writer.flush()
			train_loop.set_postfix(H_real=H_reals / (idx + 1), H_fake=H_fakes / (idx + 1))
		#end TRAINING LOOP
		
		save_image(fake_scenery * 0.5 + 0.5, os.path.join(save_folder, f"scenery_final.png"))
		save_image(fake_pixel * 0.5 + 0.5, os.path.join(save_folder, f"pixel_final.png"))
		if epoch % checkpoint_interval == 0:
			torch.save({
					'attn_config': attn_config,
					'epoch': epoch,
					'G_live_state_dict': G_live.state_dict(),
					'G_pix_state_dict': G_pix.state_dict(),
					'D_live_state_dict': D_live.state_dict(),
					'D_pix_state_dict': D_pix.state_dict(),
					'G_optimizer_state_dict': G_optimizer.state_dict(),
					'D_optimizer_state_dict': D_optimizer.state_dict(),
					'G_scaler_state_dict': G_scaler.state_dict(),
					'D_scaler_state_dict': D_scaler.state_dict(),
			}, f'checkpoints/checkpoint_{epoch}.pth')
	#end TRAINING EPOCHS
	torch.save({
		'attn_config': attn_config,
		'epoch': epoch,
		'G_live_state_dict': G_live.state_dict(),
		'G_pix_state_dict': G_pix.state_dict(),
		'D_live_state_dict': D_live.state_dict(),
		'D_pix_state_dict': D_pix.state_dict(),
		'G_optimizer_state_dict': G_optimizer.state_dict(),
		'D_optimizer_state_dict': D_optimizer.state_dict(),
		'G_scaler_state_dict': G_scaler.state_dict(),
		'D_scaler_state_dict': D_scaler.state_dict(),
	}, f'train_result.pth')
				
if __name__ == '__main__':
	#data paths
	LAMBDA_CYCLE = 10
	LAMBDA_GEN = 1
	LAMBDA_IDENTITY = 0.5
	BATCH_SIZE = 1
	NUM_EPOCHS = 100
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	Train_Dir_live="scenery/"
	Train_Dir_pix="pixel/"
	ckpt_path = "checkpoints/checkpoint_75.pth"
	if ckpt_path is None:
		attn_config = attnConfig(attn_type='cross_attn')
	
	#setting up dataset
	dataset = PixelSceneryDataset(
			root_scenery=Train_Dir_live,
			root_pixel=Train_Dir_pix,
			
	)

	# with open('train_subset.pkl', 'rb') as f:
	# 		train_indices = pickle.load(f)

	# with open('val_subset.pkl', 'rb') as f:
	# 		val_indices = pickle.load(f)

	# train_dataset = Subset(dataset, train_indices)
	# val_dataset = Subset(dataset, val_indices)

	train_dataset = dataset

	#setting up dataloader
	train_loader = DataLoader(
			train_dataset,
			batch_size=BATCH_SIZE,
			shuffle=True,
			num_workers=0,
			pin_memory=True,
	)

	# val_loader = DataLoader(
	# 		val_dataset,
	# 		batch_size=1,
	# 		shuffle=False,
	# 		pin_memory=True,
	# )
	
	val_loader = None

	if ckpt_path is None:
		#setting up scalers to prevent minor changes from defaulting to 0
		G_scaler = torch.cuda.amp.GradScaler()
		D_scaler = torch.cuda.amp.GradScaler()

		

		#setting up discriminators and generators
		G_live = gen_with_attn(attn_config).to(device)
		D_live = discriminator().to(device)
		G_pix = gen_with_attn(attn_config).to(device)
		D_pix = discriminator().to(device)
		
		#setting up optimizers
		G_optimizer = optim.Adam(list(G_live.parameters()) + list(G_pix.parameters()), lr=0.0002, betas=[0.5,0.999])
		D_optimizer = optim.Adam(list(D_live.parameters()) + list(D_pix.parameters()), lr=0.0002, betas=[0.5,0.999])
		start_epoch = 0
	else:
		attn_config, start_epoch, G_live, G_pix, D_live, D_pix, G_optimizer, D_optimizer, G_scaler, D_scaler = get_checkpoint(ckpt_path,device)

	os.makedirs('checkpoints', exist_ok=True)
	with open('checkpoints/opt.txt', 'w', encoding='utf-8') as f:
		f.write("num_epochs: 100\n")
		f.write(str(attn_config))
		f.write('\n')
		f.write('-'*10+"Generator arch"+'-'*10)
		f.write('\n')
		f.write(str(summary(G_live, (3,256,256))))
	
	#initiate training
	print("begin training")
	train_func(G_live, G_pix, D_live, D_pix, train_loader, val_loader, G_optimizer, D_optimizer, G_scaler, D_scaler, start_epoch, numEpochs=NUM_EPOCHS)

