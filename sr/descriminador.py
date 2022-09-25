from .utils import *
from .libs import *

def discriminator_block(x_in, num_filters, strides=1, batchnorm=True, momentum=0.8):
    x = Conv2D(num_filters, kernel_size=3, strides=strides, padding='same')(x_in)
    if batchnorm:
        x = BatchNormalization(momentum=momentum)(x)
    return LeakyReLU(alpha=0.2)(x)


def discriminator(num_filters=64):
    x_in = Input(shape=(256, 256, 3))
    x = Lambda(normalize_m11)(x_in)

    x = discriminator_block(x, num_filters, batchnorm=False)
    x = discriminator_block(x, num_filters, strides=2)

    x = discriminator_block(x, num_filters * 2)
    x = discriminator_block(x, num_filters * 2, strides=2)

    x = discriminator_block(x, num_filters * 4)
    x = discriminator_block(x, num_filters * 4, strides=2)

    # x = discriminator_block(x, num_filters * 8)
    # x = discriminator_block(x, num_filters * 8, strides=2)
    np.ones((3,3))
    x = Flatten()(x)

    x = Dense(64)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(1, activation='sigmoid')(x)
    # x = Dense(1, activation='softmax')(x)

    return Model(x_in, x)
class descriminador():
    def __init__(self, loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)):
        self.loss = loss

    def downsample(self, filters, size, apply_batchnorm=True):
        # initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                    # kernel_initializer=initializer, 
                                    use_bias=False))

        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())

        result.add(tf.keras.layers.LeakyReLU())

        return result
    def Discriminator(self):
        # initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
        tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

        # x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

        # down1 = self.downsample(64, 3, False)(x)  # (batch_size, 128, 128, 64)
        down1 = self.downsample(64, 3, False)(inp)  # (batch_size, 128, 128, 64)
        down2 = self.downsample(128, 3)(down1)  # (batch_size, 64, 64, 128)
        down3 = self.downsample(256, 3)(down2)  # (batch_size, 32, 32, 256)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(512, 3, strides=1,
                                        # kernel_initializer=initializer,
                                        use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

        last = tf.keras.layers.Conv2D(1, 3, strides=1,
                                        # kernel_initializer=initializer
                                        )(zero_pad2)  # (batch_size, 30, 30, 1)

        # return tf.keras.Model(inputs=inp, outputs=last)
        return tf.keras.Model(inputs=[inp, tar], outputs=last)


    def loss_discriminator(self, disc_generated_output, disc_real_output):

        real_loss = self.loss(tf.ones_like(disc_real_output), disc_real_output)

        generated_loss = self.loss(tf.zeros_like(disc_generated_output), disc_generated_output)
        

        total_disc_loss = real_loss + generated_loss
        # print(real_loss, generated_loss, total_disc_loss)
        # print(tf.shape(real_loss), tf.shape(generated_loss), tf.shape(total_disc_loss))

        return total_disc_loss

def __main__():
    pass
if __name__ == "__main__":
    pass