from multiprocessing import pool
# from unicodedata import name
from .utils import *
from .libs import *

def upsample(x_in, num_filters):
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
    x = Lambda(pixel_shuffle(scale=2))(x)
    return PReLU(shared_axes=[1, 2])(x)



class FCN32(tf.keras.models.Model):
    
    def __init__(self, n_classes=2, activation=tf.keras.activations.softmax):
        super(FCN32, self).__init__()
        self.n_classes = n_classes
        self.feature_extractor = None
        self.activation = activation
        self.vgg19 = None
 
        self.convT = tf.keras.layers.Conv2DTranspose(filters=self.n_classes, kernel_size=(32, 32), use_bias=False, strides=(32,32), padding='same')
        self.final = tf.keras.layers.Conv2D(filters=self.n_classes, kernel_size=(8,8), activation=activation, padding='same')
    def get_vgg19(self):
        if self.vgg19 == None:
            self.vgg19 = tf.keras.applications.VGG19(include_top=False,
                                                weights='imagenet',
                                                input_shape=(None,None,3))
        return self.vgg19

    def compute_output_shape(self, input_shape):
        self.inputShape = input_shape
        return (input_shape[0],input_shape[1],input_shape[2], self.n_classes)
    
    def get_feature_extractor(self):
        if self.feature_extractor == None:
            self.feature_extractor = tf.keras.models.Model(
                inputs=self.get_vgg19().input,
                outputs=[self.get_vgg19().get_layer('block1_pool').output,
                        self.get_vgg19().get_layer('block2_pool').output,
                        self.get_vgg19().get_layer('block3_pool').output,
                        self.get_vgg19().get_layer('block4_pool').output,
                        self.get_vgg19().get_layer('block5_pool').output])
        return self.feature_extractor
        

    def call(self, inputs):


        pool1, pool2, pool3, pool4, pool5 = self.feature_extractor()(inputs)     
        x = self.convT(pool5)
        x = self.final(x)
     
        return x

class Gerador_UNet():
    def __init__(self, LAMBDA=100, 
                    activation=tf.keras.activations.sigmoid, 
                    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)):
        self.loss = loss 
        self.LAMBDA = LAMBDA
        self.activation = activation
        self.inputs = None
        self.down_stack = []
        self.up_stack = []
        self.last = None
        
    def upsample(self, filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, 2, strides=1,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False))
        result.add(
            tf.keras.layers.Conv2D(filters, 3, strides=1, padding='same',
                                    kernel_initializer=initializer, 
                                    activation=tf.keras.activations.relu,
                                    use_bias=False))
        result.add(
            tf.keras.layers.Conv2D(filters, 3, strides=1, padding='same',
                                    kernel_initializer=initializer, 
                                    activation=tf.keras.activations.relu,
                                    use_bias=False))

        # result.add(tf.keras.layers.BatchNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))

        # result.add(tf.keras.layers.ReLU())

        return result

    def downsample(self, filters, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()        

        result.add(
            tf.keras.layers.Conv2D(filters, 3, strides=1, padding='same',
                                    kernel_initializer=initializer)) # use_bias=False
        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())
        result.add(ReLU())

        result.add(
            tf.keras.layers.Conv2D(filters, 3, strides=1, padding='same',
                                    kernel_initializer=initializer)) # use_bias=False
        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())
        result.add(ReLU())

        result.add(
            tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=1, padding='same',
                                    )) # use_bias=False
                                    
        result.add(
            tf.keras.layers.Dropout(0.3)) 

        return result
    
    def downsample_modify(self, filters):
        initializer = tf.random_normal_initializer(0., 0.02)

        # result = tf.keras.Sequential()
        def func(x):
            f = Conv2D(filters, 3, strides=1, padding='same',
                                    kernel_initializer=initializer,
                                    activation=relu)(x) # use_bias=False
        
            f = Conv2D(filters, 3, strides=1, padding='same',
                                    kernel_initializer=initializer,
                                    activation=relu)(f) # use_bias=False       

        
            p = MaxPooling2D(pool_size=(2,2), strides=1, padding='same')(f) # use_bias=False
            # p = Dropout(0.3)(p)
            return f, p 

        return func

    def generator(self):
        """
            down_stack = [
                self.downsample(8, 3, apply_batchnorm=True),  # (batch_size, 256, 256, 3) vs (batch_size, 128, 128, 64)
                self.downsample(16, 3),  # (batch_size, 128, 128, 64) vs (batch_size, 64, 64, 128)
                self.downsample(32, 3),  # (batch_size, 64, 64, 128) vs (batch_size, 32, 32, 256)
                self.downsample(64, 3),  # (batch_size, 32, 32, 256) vs (batch_size, 16, 16, 512)
                self.downsample(64, 3),  # (batch_size, 16, 16, 512) vs (batch_size, 8, 8, 512)
            ]

            up_stack = [
                self.upsample(64, 3),  # (batch_size, 8, 8, 64) vs (batch_size, 16, 16, 1024)
                self.upsample(32, 3),  # (batch_size, 16, 16, 64) vs (batch_size, 32, 32, 512)
                self.upsample(16, 3),  # (batch_size, 32, 32, 64) vs (batch_size, 64, 64, 256)
                self.upsample(8, 3),  # (batch_size, 64, 64, 64) vs (batch_size, 128, 128, 128)
            ]

            down_stack = [
                self.downsample(64, 3, apply_batchnorm=True),  # (batch_size, 256, 256, 3) vs (batch_size, 128, 128, 64)
                self.downsample(64*2, 3),  # (batch_size, 128, 128, 64) vs (batch_size, 64, 64, 128)
                self.downsample(64*4, 3),  # (batch_size, 64, 64, 128) vs (batch_size, 32, 32, 256)
                self.downsample(64*8, 3),  # (batch_size, 32, 32, 256) vs (batch_size, 16, 16, 512)
                self.downsample(64*16, 3),  # (batch_size, 16, 16, 512) vs (batch_size, 8, 8, 512)
            ]

            up_stack = [
                self.upsample(64*8, 3),  # (batch_size, 8, 8, 64) vs (batch_size, 16, 16, 1024)
                self.upsample(64*4, 3),  # (batch_size, 16, 16, 64) vs (batch_size, 32, 32, 512)
                self.upsample(64*2, 3),  # (batch_size, 32, 32, 64) vs (batch_size, 64, 64, 256)
                self.upsample(64, 3),  # (batch_size, 64, 64, 64) vs (batch_size, 128, 128, 128)
            ]
        """
        self.inputs = tf.keras.layers.Input(shape=[None, None, 1])
        # inputs = tf.keras.layers.Input(shape=[256, 256, 3])

        self.down_stack = [
            self.downsample(64),  # (batch_size, 256, 256, 3) vs (batch_size, 128, 128, 64)
            self.downsample(64*2),  # (batch_size, 128, 128, 64) vs (batch_size, 64, 64, 128)
            self.downsample(64*4),  # (batch_size, 64, 64, 128) vs (batch_size, 32, 32, 256)
            self.downsample(64*8),  # (batch_size, 32, 32, 256) vs (batch_size, 16, 16, 512)
            self.downsample(64*16),  # (batch_size, 16, 16, 512) vs (batch_size, 8, 8, 512)
        ]

        self.up_stack = [
            self.upsample(64*8, 3),  # (batch_size, 8, 8, 64) vs (batch_size, 16, 16, 1024)
            self.upsample(64*4, 3),  # (batch_size, 16, 16, 64) vs (batch_size, 32, 32, 512)
            self.upsample(64*2, 3),  # (batch_size, 32, 32, 64) vs (batch_size, 64, 64, 256)
            self.upsample(64, 3),  # (batch_size, 64, 64, 64) vs (batch_size, 128, 128, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        self.last = tf.keras.layers.Conv2D(1, 1,
                                        strides=1,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        activation=self.activation)  # (batch_size, 256, 256, 3) tanh # 'softmax'
                                        # activation=tf.keras.activations.softmax)  # (batch_size, 256, 256, 3) tanh # 'softmax'

        x = self.inputs
        # Downsampling through the model
        skips = []
        for down in self.down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            x = Concatenate()([x, skip])

        x = self.last(x)

        return Model(inputs=self.inputs, outputs=x)
    
    def generator_modify(self):
        """
            down_stack = [
                self.downsample(8, 3, apply_batchnorm=True),  # (batch_size, 256, 256, 3) vs (batch_size, 128, 128, 64)
                self.downsample(16, 3),  # (batch_size, 128, 128, 64) vs (batch_size, 64, 64, 128)
                self.downsample(32, 3),  # (batch_size, 64, 64, 128) vs (batch_size, 32, 32, 256)
                self.downsample(64, 3),  # (batch_size, 32, 32, 256) vs (batch_size, 16, 16, 512)
                self.downsample(64, 3),  # (batch_size, 16, 16, 512) vs (batch_size, 8, 8, 512)
            ]

            up_stack = [
                self.upsample(64, 3),  # (batch_size, 8, 8, 64) vs (batch_size, 16, 16, 1024)
                self.upsample(32, 3),  # (batch_size, 16, 16, 64) vs (batch_size, 32, 32, 512)
                self.upsample(16, 3),  # (batch_size, 32, 32, 64) vs (batch_size, 64, 64, 256)
                self.upsample(8, 3),  # (batch_size, 64, 64, 64) vs (batch_size, 128, 128, 128)
            ]

            down_stack = [
                self.downsample(64, 3, apply_batchnorm=True),  # (batch_size, 256, 256, 3) vs (batch_size, 128, 128, 64)
                self.downsample(64*2, 3),  # (batch_size, 128, 128, 64) vs (batch_size, 64, 64, 128)
                self.downsample(64*4, 3),  # (batch_size, 64, 64, 128) vs (batch_size, 32, 32, 256)
                self.downsample(64*8, 3),  # (batch_size, 32, 32, 256) vs (batch_size, 16, 16, 512)
                self.downsample(64*16, 3),  # (batch_size, 16, 16, 512) vs (batch_size, 8, 8, 512)
            ]

            up_stack = [
                self.upsample(64*8, 3),  # (batch_size, 8, 8, 64) vs (batch_size, 16, 16, 1024)
                self.upsample(64*4, 3),  # (batch_size, 16, 16, 64) vs (batch_size, 32, 32, 512)
                self.upsample(64*2, 3),  # (batch_size, 32, 32, 64) vs (batch_size, 64, 64, 256)
                self.upsample(64, 3),  # (batch_size, 64, 64, 64) vs (batch_size, 128, 128, 128)
            ]
        """
        inputs = tf.keras.layers.Input(shape=[None, None, 1])
        # inputs = tf.keras.layers.Input(shape=[256, 256, 3])

        down_stack = [
            self.downsample_modify(64),  # (batch_size, 256, 256, 3) vs (batch_size, 128, 128, 64)
            self.downsample_modify(64*2),  # (batch_size, 128, 128, 64) vs (batch_size, 64, 64, 128)
            self.downsample_modify(64*4),  # (batch_size, 64, 64, 128) vs (batch_size, 32, 32, 256)
            self.downsample_modify(64*8),  # (batch_size, 32, 32, 256) vs (batch_size, 16, 16, 512)
            self.downsample_modify(64*16),  # (batch_size, 16, 16, 512) vs (batch_size, 8, 8, 512)
        ]

        up_stack = [
            self.upsample(64*8, 3),  # (batch_size, 8, 8, 64) vs (batch_size, 16, 16, 1024)
            self.upsample(64*4, 3),  # (batch_size, 16, 16, 64) vs (batch_size, 32, 32, 512)
            self.upsample(64*2, 3),  # (batch_size, 32, 32, 64) vs (batch_size, 64, 64, 256)
            self.upsample(64, 3),  # (batch_size, 64, 64, 64) vs (batch_size, 128, 128, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2D(2, 1,
                                        strides=1,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        activation=self.activation, 
                                        name = "last_layer")  # (batch_size, 256, 256, 3) tanh # 'softmax'

        x = inputs
        # Downsampling through the model
        skips = []
        for down in down_stack:
            f, x = down(x)
            skips.append(f)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = Concatenate()([x, skip])

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)

def __main__():
    pass
if __name__ == "__main__":
    pass

# unet para remocão de ruido de imagens medicas
# secão dois do trabalho

# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9446143
# monta apresentacão
#     se tem o git