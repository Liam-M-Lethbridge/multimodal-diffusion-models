import tensorflow as tf
from keras import layers, models


class ConvBlock(layers.Layer):
    """Convolution block class. Each block consists of a convolution layer, batch norm and relu activation twice."""
    def __init__(self, filters, kernel_size=3):
        super().__init__()
        self.conv1 = layers.Conv2D(filters, kernel_size, padding="same")
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()

        self.conv2 = layers.Conv2D(filters, kernel_size, padding="same")
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu2(x)
        return x

class EncoderBlock(layers.Layer):
    """Encoder block class. Consists of convolution block and pooling layer to shrink the image but increase the channels."""
    def __init__(self, filters):
        super().__init__()
        self.conv = ConvBlock(filters)
        self.pool = layers.MaxPooling2D((2, 2))

    def call(self, x, training=False):
        f = self.conv(x, training=training)
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

    def call(self, x, skip, training=False):
        x = self.up(x)
        x = self.concat([x, skip])
        x = self.conv(x, training=training)
        return x


class TimeEmbedding(tf.keras.layers.Layer):
    """Time embedding class. Used for embedding time step information into the diffusion model"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def call(self, t):
        half = self.dim // 2
        freqs = tf.exp(
            -tf.math.log(10000.0) * tf.range(half, dtype=tf.float32) / half
        )
        args = t[:, None] * freqs[None]
        emb = tf.concat([tf.sin(args), tf.cos(args)], axis=-1)
        return emb

class CondEncoder(tf.keras.layers.Layer):
    def __init__(self, base=64):
        super().__init__()
        self.conv1 = ConvBlock(base)
        self.conv2 = ConvBlock(base * 2)
        self.conv3 = ConvBlock(base * 4)

    def call(self, x, training=False):
        x = self.conv1(x, training)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = self.conv2(x, training)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = self.conv3(x, training)
        return x  # (B, H/4, W/4, C)
    
class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = tf.keras.layers.LayerNormalization()
        self.attn = tf.keras.layers.MultiHeadAttention(
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

    def call(self, x, skip, cond, training=False):
        x = self.up(x)
        x = tf.concat([x, skip], axis=-1)
        x = self.conv(x, training)
        x = self.attn(x, cond, training)
        return x
