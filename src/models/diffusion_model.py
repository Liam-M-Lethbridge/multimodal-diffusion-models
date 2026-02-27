"""This file contains a class for a basic diffusion model."""
import tensorflow as tf
import keras
from model import Model
import matplotlib.pyplot as plt

class DiffusionModel(Model):
    def __init__(self):
        pass

    def initialise_model():
        pass

    def save_model():
        pass

    def load_model():
        pass

    def train_model():
        pass

def normalisation(img):
    img = img/255.0
    return img
def examples(x):
    plt.figure(figsize=(9, 9))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        img = normalisation(x[i])
        plt.imshow(img)
        plt.axis('off')
    plt.show()
if __name__ == "__main__":

    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print(X_train.shape)
    examples(X_train)