from models.UNet import DiffUNetCrossAttention
from training.trainUNet import train_model

if __name__== "__main__":
    unet = DiffUNetCrossAttention("Diffusion_Unet")
    train_model(unet, 20, 1, 4)