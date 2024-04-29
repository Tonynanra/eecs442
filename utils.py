from generator import generator, gen_with_attn, attnConfig
from Discriminator_442 import discriminator
import torch
from torch import optim
def get_checkpoint(resume_from,device):
	checkpoint = torch.load(resume_from)
	attn_config = checkpoint['attn_config']
	start_epoch = checkpoint['epoch']
	G_live = gen_with_attn(attn_config).to(device)
	G_pix = gen_with_attn(attn_config).to(device)
	D_live = discriminator().to(device)
	D_pix = discriminator().to(device)
	G_optimizer = optim.Adam(list(G_live.parameters()) + list(G_pix.parameters()), lr=2e-4, betas=(0.5, 0.999))
	D_optimizer = optim.Adam(list(D_live.parameters()) + list(D_pix.parameters()), lr=2e-4, betas=(0.5, 0.999))
	G_scaler = torch.cuda.amp.GradScaler()
	D_scaler = torch.cuda.amp.GradScaler()
	G_live.load_state_dict(checkpoint['G_live_state_dict'])
	G_pix.load_state_dict(checkpoint['G_pix_state_dict'])
	D_live.load_state_dict(checkpoint['D_live_state_dict'])
	D_pix.load_state_dict(checkpoint['D_pix_state_dict'])
	G_optimizer.load_state_dict(checkpoint['G_optimizer_state_dict'])
	D_optimizer.load_state_dict(checkpoint['D_optimizer_state_dict'])
	G_scaler.load_state_dict(checkpoint['G_scaler_state_dict'])
	D_scaler.load_state_dict(checkpoint['D_scaler_state_dict'])

	return attn_config, start_epoch, G_live, G_pix, D_live, D_pix, G_optimizer, D_optimizer, G_scaler, D_scaler