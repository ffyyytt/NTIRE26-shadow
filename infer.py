import albumentations as A

import glob
import lpips
import torch
import torchvision

import numpy as np
import albumentations as A
import matplotlib.pyplot as plt

from tqdm import *
from PIL import Image
from skimage.color import rgb2lab
from fvcore.nn import FlopCountAnalysis

class Dataset(torch.utils.data.Dataset):
    def __init__(self, shadow_files):
        self.shadow_files = shadow_files
        self.torchTrans = torchvision.transforms.Compose([torchvision.transforms.Resize([720, 960]),
                                                          torchvision.transforms.ToTensor(),
                                                          torchvision.transforms.Normalize(mean=[0.5], std=[0.5])])
        
    def __len__(self):
        return len(self.shadow_files)

    def __getitem__(self, idx):
        shadow_image = Image.open(self.shadow_files[idx]).convert("RGB")
        return {"idx": idx, 
                "shadow_image": self.torchTrans(shadow_image)
        }
        
        
def denormalize(tensor):
    tensor = tensor*0.5 + 0.5
    tensor = tensor.permute(0, 2, 3, 1).cpu().numpy()
    return np.rint(tensor * 255).astype(np.uint8)

class ModelFactory(torch.nn.Module):
    def __init__(self, vae, unet):
        super(ModelFactory, self).__init__()
        self.vae = vae
        self.unet = unet

    def rgb_to_latent(self, images):
        latents = self.vae.encode(images).latent_dist.sample() * 0.18215
        return latents

    def latent_to_rgb(self, latents):
        latents = latents / 0.18215
        imgs = self.vae.decode(latents).sample
        return imgs

    def get_output(self, shadow_images, timestep=0):
        shadow_latents = self.rgb_to_latent(shadow_images)
        prompt_embeds = torch.zeros([shadow_latents.shape[0], 1, 1024], device=shadow_latents.device, dtype=shadow_latents.dtype)
        residual = self.unet(shadow_latents,
                             timestep=torch.tensor([timestep], device=shadow_latents.device),
                             encoder_hidden_states=prompt_embeds).sample
        return self.latent_to_rgb(residual)

    def forward(self, shadow_images, timestep=1, infer=False):
        residual = self.get_output(shadow_images, timestep=timestep)
        output_images = (residual + shadow_images).tanh()

        # g = torch.sigmoid(128.0 * (output_images - shadow_images))
        # output_images = g * output_images + (1 - g) * shadow_images
        return output_images
        
        
shadow_files = glob.glob("dataset/test/*.*")
valid_dataset = torch.utils.data.DataLoader(Dataset(shadow_files = shadow_files), batch_size=1, shuffle=False, num_workers=28)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model0 = torch.load("models/ntire0.pth", weights_only=False).to(device)
model1 = torch.load("models/ntire1.pth", weights_only=False).to(device)
model2 = torch.load("models/ntire2.pth", weights_only=False).to(device)
model3 = torch.load("models/ntire3.pth", weights_only=False).to(device)

model0.eval()
model1.eval()
model2.eval()
model3.eval()

for batch in tqdm(valid_dataset):
    idx = batch["idx"]
    shadow_images = batch["shadow_image"].to(device)
    with torch.no_grad():
        remove_images0 = model0(shadow_images)
        remove_images1 = model1(shadow_images)
        remove_images2 = model2(shadow_images)
        remove_images3 = model3(shadow_images)

    remove_images = (remove_images0 + remove_images1 + remove_images2 + remove_images3)/4
    image = Image.fromarray(denormalize(remove_images)[0])
    image.save("output/" + shadow_files[idx].split("/")[-1].replace(".jpg", ".png"))