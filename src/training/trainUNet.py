"""Generic UNet training for semantic segmentation of images."""
from models.UNet import UNet
from misc.dataset import getBatchInOrder, getRandomBatch




def trainModel(nEpochs: int = 100, batchSize: int = 1000):
    RGBBatch, SEGBatch = getRandomBatch(batchSize, "UK")

    model = UNet()


