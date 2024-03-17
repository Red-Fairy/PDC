import torch
import numpy as np
from torch import nn

class PositionalEncoding(nn.Module):
	def __init__(self, min_deg=0, max_deg=5):
		super(PositionalEncoding, self).__init__()
		self.min_deg = min_deg
		self.max_deg = max_deg
		self.scales = torch.tensor([2 ** i for i in range(min_deg, max_deg)])

	def forward(self, x):
		# x: B*3
		x_ = x
		shape = list(x.shape[:-1]) + [-1]
		x_enc = (x[..., None, :] * self.scales[:, None].to(x.device)).reshape(shape)
		x_enc = torch.cat((x_enc, x_enc + 0.5 * torch.pi), -1)

		# PE
		x_ret = torch.sin(x_enc)
		x_ret = torch.cat([x_ret, x_], dim=-1) # B*(6*(max_deg-min_deg)+3)
		return x_ret
		
class BBoxModel(nn.Module):
	def __init__(self, output_dim, hidden_dim=128, min_deg=0, max_deg=5):
		'''
		positional encoding + 2-layer MLP
		'''
		super(BBoxModel, self).__init__()
		self.output_dim = output_dim
		self.hidden_dim = hidden_dim
		self.min_deg = min_deg
		self.max_deg = max_deg
		self.pe = PositionalEncoding(min_deg, max_deg)
		input_dim = 6 * (max_deg - min_deg) + 3

		self.encoder = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, output_dim)
		)

		self.null_cond = nn.Parameter(torch.randn(output_dim))

	def forward(self, x: torch.Tensor=None, uncond=False):
		# x: B*3
		
		if not uncond:
			# 1) first normalize the input, such that the largest value for each instance is 1
			x = x / torch.max(x, dim=1, keepdim=True)[0]
			x = self.pe(x) # B*(6*(max_deg-min_deg)+3)
			x = self.encoder(x)
			return x
		
		else:
			return self.null_cond


