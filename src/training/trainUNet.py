"""Generic UNet training for semantic segmentation of images."""
from models.UNet import DiffUNetCrossAttention
from misc.dataset import getBatchInOrder, getRandomBatch
import tensorflow as tf
import keras
from misc.pipeline import TrainingPipeline

def train_model(model, epochs=50, checkpoint_rate = 5, batch_size = 32):
    optimiser = keras.optimizers.Adam(learning_rate=1e-4)
    loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)
    pipeline = TrainingPipeline()
    n_batches = len(pipeline.filenames)//batch_size
    print(f"epochs: {epochs}, batches: {n_batches}")
    for epoch in range(epochs):
        for batch in range(n_batches):
            noisy_seg_map, rgb, noise, t = pipeline.get_batch(batch_size)
            loss = train_on_batch(model, optimiser, noisy_seg_map, rgb, noise, t)

            print(f"Epoch {epoch+1}, batch:{batch} loss: {loss.numpy()}")
        if checkpoint_rate == 0:
            if epoch + 1 == epochs:
                model.checkpoint(epochs+1)
        elif (epoch+1)%checkpoint_rate == 0:
            model.checkpoint(epoch+1) 




@tf.function
def train_on_batch(model, optimiser, noisy_seg_map, rgb, noise, t):

    with tf.GradientTape() as tape:
        pred_noise = model(noisy_seg_map, t, rgb, training=True)
        loss = tf.reduce_mean(tf.square(noise - pred_noise))

    gradients = tape.gradient(loss, model.trainable_variables)
    optimiser.apply_gradients(zip(gradients, model.trainable_variables))

    return loss

