""" 
    Reference:
        - https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/encoders/modules.py
        - https://github.com/openai/CLIP
"""

# import kornia
from einops import rearrange, repeat
from torchvision import transforms

import torch
import torch.nn as nn
from PIL import Image

# from external.clip import clip
import clip

class CLIPImageEncoder(nn.Module):
    def __init__(
            self,
            model="ViT-B/32",
            jit=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            antialias=False,
        ):
        super().__init__()
        self.model, _ = clip.load(name=model, device=device, jit=jit)

        # self.model, self.preprocess = clip.load(name=model, device=device, jit=jit)
        self.model = self.model.float() # turns out this is important...

        self.antialias = antialias

        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

    def preprocess(self, x):
        # normalize to [0,1]

        x = transforms.Resize((224, 224), antialias=self.antialias, interpolation=Image.BICUBIC)(x)
        x = (x + 1.) / 2.
        # renormalize according to clip
        x = transforms.Normalize(mean=self.mean, std=self.std)(x)
        return x

    def forward(self, x):
        # x is assumed to be in range [-1,1]
        return self.model.encode_image(self.preprocess(x))
    
if __name__ == "__main__":
    # Test CLIPImageEncoder
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.autograd import Variable

    # Create a random image
    image = torch.rand(1, 3, 224, 224)
    image = Variable(image, requires_grad=True)

    # Create the CLIPImageEncoder
    clip_image_encoder = CLIPImageEncoder()

    # Forward pass
    output = clip_image_encoder(image)

    # Print output shape
    print(f"Output shape: {output.shape}")
