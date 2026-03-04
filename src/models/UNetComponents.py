import tensorflow as tf
from keras import layers, models

def timestep_embedding(t, dim):
    """
    Create sinusoidal timestep embeddings.
    t: (B,) int32
    dim: embedding dimension
    """

    half_dim = dim // 2
    freqs = tf.exp(
        -tf.math.log(10000.0) * tf.range(0, half_dim, dtype=tf.float32) / half_dim
    )

    t = tf.cast(t, tf.float32)
    args = tf.expand_dims(t, 1) * tf.expand_dims(freqs, 0)

    embedding = tf.concat([tf.sin(args), tf.cos(args)], axis=-1)

    return embedding


class ConvBlock(layers.Layer):
    """Convolution block class. Each block consists of a convolution layer, batch norm, time embedding, and relu activation twice."""
    def __init__(self, filters, kernel_size=3):
        super().__init__()

        self.filters = filters

        self.conv1 = layers.Conv2D(filters, kernel_size, padding="same")
        self.norm1 = layers.GroupNormalization()

        self.time_embedder = layers.Dense(filters)

        self.conv2 = layers.Conv2D(filters, kernel_size, padding="same")
        self.norm2 = layers.GroupNormalization()

        self.activation = layers.Activation("swish")

        self.projection = layers.Conv2D(filters, 1, padding="same")

    def call(self, x, t, training=False):
        input_tensor = x

        # need to project if the dimensions are not the same
        if input_tensor.shape[-1] != self.filters:
            input_tensor = self.projection(input_tensor)

        # first convolution
        x = self.conv1(x)
        x = self.norm1(x, training=training)
        x = self.activation(x)

        # inject time embedding
        te = timestep_embedding(t, self.filters)
        te = self.time_embedder(te)  
        te = te[:, None, None, :]     
        x = x + te
        
        # second convolution
        x = self.conv2(x)
        x = self.norm2(x, training=training)
        x = self.activation(x)

        # residual connection
        return x + input_tensor

class EncoderBlock(layers.Layer):
    """Encoder block class. Consists of convolution block and pooling layer to shrink the image but increase the channels."""
    def __init__(self, filters):
        super().__init__()
        self.conv = ConvBlock(filters)
        self.pool = layers.MaxPooling2D((2, 2))

    def call(self, x, t, training=False):
        f = self.conv(x, t, training=training)
        p = self.pool(f)
        return f, p

class DecoderBlock(layers.Layer):
    """Decoder block class. Concatenates the skip path from related encoder blocks with up-conv output."""
    def __init__(self, filters):
        super().__init__()
        self.up = layers.Conv2DTranspose(
            filters, (2, 2), strides=2, padding="same"
        )
        self.concat = layers.Concatenate()
        self.conv = ConvBlock(filters)

    def call(self, x, skip, t, training=False):
        x = self.up(x)
        x = self.concat([x, skip])
        x = self.conv(x, t, training=training)
        return x




class CondEncoder(tf.keras.layers.Layer):
    def __init__(self, base=64):
        super().__init__()
        self.conv1 = ConvBlock(base)
        self.conv2 = ConvBlock(base * 2)
        self.conv3 = ConvBlock(base * 4)

    def call(self, x, t, training=False):
        x = self.conv1(x, t, training=training)
        x = layers.MaxPooling2D()(x)
        x = self.conv2(x, t, training=training)
        x = layers.MaxPooling2D()(x)
        x = self.conv3(x, t, training=training)
        return x  
    
class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = layers.LayerNormalization()
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=channels // num_heads
        )

    def call(self, x, cond, training=False):
        """
        Args:
            x: queries.
            cond: keys and values.
        """
        b, h, w, c = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]

        x_norm = self.norm(x)

        q = tf.reshape(x_norm, (b, h * w, c))
        kv = tf.reshape(cond, (b, -1, c))

        attn_out = self.attn(q, kv, kv, training=training)
        attn_out = tf.reshape(attn_out, (b, h, w, c))

        return x + attn_out

class AttentionDecoderBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super().__init__()
        self.up = layers.Conv2DTranspose(filters, 2, strides=2, padding="same")
        self.conv = ConvBlock(filters)
        self.attn = CrossAttention(filters)

    def call(self, x, skip, cond, t, training=False):
        x = self.up(x)
        x = tf.concat([x, skip], axis=-1)
        x = self.conv(x, t, training=training)
        x = self.attn(x, cond, training=training)
        return x
