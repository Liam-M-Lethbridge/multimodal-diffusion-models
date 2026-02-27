import tensorflow as tf
from keras import layers, models
from UNetComponents import ConvBlock, EncoderBlock, DecoderBlock, AttentionDecoderBlock


class UNet(tf.keras.Model):
    """U-net model class (RGB -> seg). Combines encoder and decoders in U-net architecture and performs skips"""
    def __init__(self, num_classes=9):
        super().__init__()
        self.num_classes = num_classes

        # Encoder
        self.encoder1 = EncoderBlock(64)
        self.encoder2 = EncoderBlock(128)
        self.encoder3 = EncoderBlock(256)

        # Bottleneck
        self.bottleneck = ConvBlock(512)

        # Decoder
        self.decoder1 = DecoderBlock(256)
        self.decoder2 = DecoderBlock(128)
        self.decoder3 = DecoderBlock(64)

        # Output layer
        self.classifier = layers.Conv2D(
            num_classes, (1, 1), padding="same", activation="softmax"
        )

    def call(self, x, training=False):
        # Encoder
        skip1, e1 = self.encoder1(x, training=training)
        skip2, e2 = self.encoder2(e1, training=training)
        skip3, e3 = self.encoder3(e2, training=training)

        # Bottleneck
        b = self.bottleneck(e3, training=training)

        # Decoder
        d1 = self.decoder1(b, skip3, training=training)
        d2 = self.decoder2(d1, skip2, training=training)
        d3 = self.decoder3(d2, skip1, training=training)

        return self.classifier(d3)



class DiffUNet(tf.keras.Model):
    """U-net model class for diffusion (RGB -> seg). Combines encoder and decoders in U-net architecture and performs skips.
    """
    def __init__(self, num_classes=9):
        super().__init__()
        self.num_classes = num_classes


        # Encoder
        self.encoder1 = EncoderBlock(64, input_channels = num_classes)
        self.encoder2 = EncoderBlock(128)
        self.encoder3 = EncoderBlock(256)

        # Bottleneck
        self.bottleneck = ConvBlock(512)

        # Decoder
        self.decoder1 = DecoderBlock(256)
        self.decoder2 = DecoderBlock(128)
        self.decoder3 = DecoderBlock(64)

        # Output layer
        self.outputLayer = layers.Conv2D(
            num_classes, (1, 1), padding="same", activation="softmax"
        )


    def call(self, x, t, training=False):


        # Encoder
        skip1, e1 = self.encoder1(x, t, training=training)
        skip2, e2 = self.encoder2(e1, t, training=training)
        skip3, e3 = self.encoder3(e2, t, training=training)

        # Bottleneck
        b = self.bottleneck(e3, t, training=training)

        # Decoder
        d1 = self.decoder1(b, skip3, t, training=training)
        d2 = self.decoder2(d1, skip2, t, training=training)
        d3 = self.decoder3(d2, skip1, t, training=training)

        return self.outputlayer(d3)



class DiffUNetCrossAttention(tf.keras.Model):
    """U-net model class for diffusion using cross attention. Combines encoder and decoders in U-net architecture and performs skips. 
"""
    def __init__(self, input_channels=3, num_classes=9):
        super().__init__()
        self.num_classes = num_classes

        # Encoder
        self.encoder1 = EncoderBlock(64)
        self.encoder2 = EncoderBlock(128)
        self.encoder3 = EncoderBlock(256)

        # Bottleneck
        self.bottleneck = ConvBlock(512)
        
        # Decoder
        self.decoder1 = AttentionDecoderBlock(256)
        self.decoder2 = AttentionDecoderBlock(128)
        self.decoder3 = AttentionDecoderBlock(64)

        # Output layer
        self.outputlayer = layers.Conv2D(
            num_classes, (1, 1), padding="same"
        )

    def call(self, x, t, cond, training=False):
        """Call function.
        x: the noisy logit space matrix (W x H x C).
        t: the time step.
        cond: the image conditioning on.
        training: a boolean representing the training"""


        # Encoder
        skip1, e1 = self.encoder1(x, t, training=training)
        skip2, e2 = self.encoder2(e1, t, training=training)
        skip3, e3 = self.encoder3(e2, t, training=training)

        # Bottleneck
        b = self.bottleneck(e3, t, training=training)

        # Decoder
        d1 = self.decoder1(b, skip3, cond, t, training=training)
        d2 = self.decoder2(d1, skip2, cond, t, training=training)
        d3 = self.decoder3(d2, skip1, cond, t, training=training)

        return self.outputlayer(d3)
