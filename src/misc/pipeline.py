import tensorflow as tf
import keras
import numpy as np
import os
import random

def cosine_beta_schedule(n_timesteps, s=0.008):
    """
    Cosine schedule as proposed in
    'Improved Denoising Diffusion Probabilistic Models'
    
    Args:
        n_timesteps: number of diffusion steps
        s: small offset to prevent singularities (default 0.008)
    """

    steps = n_timesteps + 1
    x = np.linspace(0, n_timesteps, steps)

    # cosine schedule for alpha_bar
    alphas_hat = np.cos(((x / n_timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_hat = alphas_hat / alphas_hat[0]

    # derive betas
    betas = 1 - (alphas_hat[1:] / alphas_hat[:-1])

    return np.clip(betas, 1e-8, 0.999)

class TrainingPipeline():
    """A class for building a training pipeline."""
    def __init__(self, beta_start = 1e-4, beta_end = 1e-2, n_timesteps = 300,  directory = "data/UK/training/", n_classes = 10):
        self.betas = cosine_beta_schedule(n_timesteps).astype(np.float32)
        self.alphas = (1.0 - self.betas).astype(np.float32)
        self.alphas_hat = np.cumprod(self.alphas).astype(np.float32)

        # convert to TF tensors
        self.alphas_hat = tf.constant(self.alphas_hat, dtype=tf.float32)
        self.n_timesteps = n_timesteps
        self.directory = directory
        self.filenames = os.listdir(directory + "SEG/")

    def get_batch(self, batch_size=32):

        # if we have run out of files, we reset the list.
        if len(self.filenames) < batch_size:
            self.filenames = os.listdir(self.directory + "SEG/")

        batch_names = random.sample(self.filenames, batch_size)

        for name in batch_names:
            self.filenames.remove(name)

        clean_segmentation_maps = []
        rgb_images = []

        rgb_dir = self.directory + "RGB/"
        seg_dir = self.directory + "SEG/"

        for file in batch_names:

            seg = np.load(seg_dir + file)  # (H,W)
            seg = seg.astype(np.int32)

            # convert to one-hot
            seg = tf.one_hot((seg//10)-1, depth=10)  # (H,W,10)

            # then convert to logits
            seg = seg*2.0 -1.0

            rgb = np.load(rgb_dir + file).astype(np.float32)

            clean_segmentation_maps.append(seg)
            rgb_images.append(rgb)

        clean_segmentation_maps = tf.stack(clean_segmentation_maps)
        clean_segmentation_maps = clean_segmentation_maps * 2.0 - 1.0
        rgb_images = tf.stack(rgb_images)

        # sample random timesteps for each image
        t = tf.random.uniform(
            shape=(batch_size,),
            minval=0,
            maxval=self.n_timesteps,
            dtype=tf.int32
        )

        # sample Gaussian noise
        noise = tf.random.normal(shape=tf.shape(clean_segmentation_maps))

        # gather alpha_hat for each t
        alpha_hat = tf.gather(self.alphas_hat, t)
        alpha_hat = tf.reshape(alpha_hat, (-1, 1, 1, 1))

        # diffusion forward equation
        noisy_seg_maps = (
            tf.sqrt(alpha_hat) * clean_segmentation_maps +
            tf.sqrt(1.0 - alpha_hat) * noise
        )

        # store for training
        self.clean_seg_maps = clean_segmentation_maps
        self.noisy_seg_maps = noisy_seg_maps
        self.rgb_images = rgb_images
        self.noise = noise
        self.timesteps = t

        return noisy_seg_maps, rgb_images, noise, t
        
    def sample(self, model, rgb):

        batch_size = tf.shape(rgb)[0]
        image = tf.random.normal((batch_size, 256, 256, 10))

        for t in reversed(range(self.n_timesteps)):

            t_tensor = tf.fill((batch_size,), t)

            pred_noise = model(image, t_tensor, rgb, training=False)

            alpha_t = self.alphas[t]
            alpha_hat_t = self.alphas_hat[t]
            beta_t = self.betas[t]

            alpha_t = tf.cast(alpha_t, tf.float32)
            alpha_hat_t = tf.cast(alpha_hat_t, tf.float32)
            beta_t = tf.cast(beta_t, tf.float32)

            # reshape scalars for broadcasting
            alpha_t = tf.reshape(alpha_t, (1,1,1,1))
            alpha_hat_t = tf.reshape(alpha_hat_t, (1,1,1,1))
            beta_t = tf.reshape(beta_t, (1,1,1,1))

            mean = (1.0 / tf.sqrt(alpha_t)) * (
                image - (beta_t / tf.sqrt(1.0 - alpha_hat_t)) * pred_noise
            )

            if t > 0:
                noise = tf.random.normal(tf.shape(image))
                image = mean + tf.sqrt(beta_t) * noise
            else:
                image = mean

        return image

