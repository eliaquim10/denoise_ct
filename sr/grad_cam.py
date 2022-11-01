from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.cm as cm
import keras

class GradCAM:
    def __init__(self, model, classIdx, layerName=None):
        # store the model, the class index used to measure the class
        # activation map, and the layer to be used when visualizing
        # the class activation map
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName
        # if the layer name is None, attempt to automatically find
        # the target output layer
        if self.layerName is None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        # attempt to find the final convolutional layer in the network
        # by looping over the layers of the network in reverse order
        for layer in reversed(self.model.layers):
            # check to see if the layer has a 4D output
            if len(layer.output_shape) == 4:
                return layer.name
        # otherwise, we could not find a 4D layer so the GradCAM
        # algorithm cannot be applied
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")


    def compute_heatmap(self, image, eps=1e-8):
        # construct our gradient model by supplying (1) the inputs
        # to our pre-trained model, (2) the output of the (presumably)
        # final 4D layer in the network, and (3) the output of the
        # softmax activations from the model
        # image = tf.expand_dims(image, axis=2)
        image = tf.expand_dims(image, axis=0)
        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output, self.model.output])

        # record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # cast the image tensor to a float-32 data type, pass the
            # image through the gradient model, and grab the loss
            # associated with the specific class index
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            
            loss = predictions[:, self.classIdx]
    
        # use automatic differentiation to compute the gradients
        grads = tape.gradient(loss, convOutputs)

        print("convOutputs", convOutputs.shape)
        print("guidedGrads", grads.shape)
        print("convOutputs", tf.reduce_sum(convOutputs))
        print("guidedGrads", tf.reduce_sum(grads))
        # compute the guided gradients
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads
        # the convolution and guided gradients have a batch dimension
        # (which we don't need) so let's grab the volume itself and
        # discard the batch
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        # compute the average of the gradient values, and using them
        # as weights, compute the ponderation of the filters with
        # respect to the weights
        # if castGrads.numpy() > 0:
        #     weights = tf.reduce_mean(guidedGrads, axis=(0,1))
        # else:
        weights = self.model.get_layer("last_layer").weights[0][0,0,-1,0]
        # print(weights)
        # cam = weights @ convOutputs
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
        
        # grab the spatial dimensions of the input image and resize
        # the output class activation map to match the input image
        # dimensions
        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))
        # normalize the heatmap such that all values lie in the range
        # [0, 1], scale the resulting values to the range [0, 255],
        # and then convert to an unsigned 8-bit integer

        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")
        # return the resulting heatmap to the calling function
        return heatmap

    def make_gradcam_heatmap(self, img, pred_index=None):
        # First, we create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        grad_model = tf.keras.models.Model(
            [self.model.inputs], [self.model.get_layer(self.layerName).output, self.model.output]
        )
        data = tf.expand_dims(img, axis=2)
        data = tf.expand_dims(data, axis=0)

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(data)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, self.classIdx]

        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
        # print(pooled_grads)

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

    def overlay_heatmap(self, heatmap, image, alpha=0.5,
                        colormap=cv2.COLORMAP_VIRIDIS):
        # apply the supplied color map to the heatmap and then
        # overlay the heatmap on the input image
        image = image.numpy()

        # image = np.clip(image, 0, 255)
        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        print("image", image.shape)
        print("heatmap", heatmap.shape)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
        heatmap = np.clip(heatmap, 0, 255)
        heatmap = cv2.applyColorMap(heatmap, colormap)
        # output = image + (heatmap * alpha)
        # return a 2-tuple of the color mapped heatmap and the output,
        # overlaid image
        return (heatmap, output)
    
    def display_gradcam(self, heatmap, image, alpha=0.4):
        # Load the original image
        image = tf.expand_dims(image, 2)
        image = tf.clip_by_value(image,0,255)

        image = cv2.cvtColor(image.numpy(), cv2.COLOR_GRAY2RGB)

        # image = keras.preprocessing.image.array_to_img(image)
        # Rescale heatmap to a range 0-255
        np.clip(heatmap, 0, 255, heatmap)
        # heatmap = np.uint8(255 * heatmap)

        # Use jet colormap to colorize heatmap

        # jet_heatmap = heatmap
        jet = cm.get_cmap("jet")

        # # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # Create an image with RGB colorized heatmap
        # self.resize(jet_heatmap, image)
        # jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
        # jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        # jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

        # Superimpose the heatmap on original image
        # print("jet_heatmap", jet_heatmap.shape, image.shape)
        superimposed_img = jet_heatmap * alpha + image
        superimposed_img = tf.clip_by_value(superimposed_img,0,255)
        # np.clip(superimposed_img, 0, 255, superimposed_img)
        # superimposed_img = cv2.cvtColor(superimposed_img.numpy(), cv2.COLOR_GRAY2RGB)

        # superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

        # Save the superimposed image
        return jet_heatmap, superimposed_img

    def resize(self, heatmap, entrada):
        return cv2.resize(heatmap, (entrada.shape[1], entrada.shape[0]))


def __main__():
    pass
if __name__ == "__main__":
    pass