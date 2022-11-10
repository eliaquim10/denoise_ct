import tensorflow as tf
import tensorflow.keras.backend as K

class DiceLoss(tf.losses.Loss):
    """Creates a criterion to measure Dice loss:
    .. math:: L(precision, recall) = 1 - (1 + \beta^2) \frac{precision \cdot recall}
        {\beta^2 \cdot precision + recall}
    The formula in terms of *Type I* and *Type II* errors:
    .. math:: L(tp, fp, fn) = \frac{(1 + \beta^2) \cdot tp} {(1 + \beta^2) \cdot fp + \beta^2 \cdot fn + fp}
    where:
         - tp - true positives;
         - fp - false positives;
         - fn - false negatives;
    Args:
        beta: Float or integer coefficient for precision and recall balance.
        class_weights: Array (``np.array``) of class weights (``len(weights) = num_classes``).
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.
        per_image: If ``True`` loss is calculated for each image in batch and then averaged,
        else loss is calculated for the whole batch.
        smooth: Value to avoid division by zero.
    Returns:
        A callable ``dice_loss`` instance. Can be used in ``model.compile(...)`` function`
        or combined with other losses.
    Example:
    .. code:: python
        loss = DiceLoss()
        model.compile('SGD', loss=loss)
    """

    def __init__(self, beta=1, class_weights=None, class_indexes=None, per_image=False, smooth=1e-5):
        super().__init__(name='dice_loss')
        self.beta = beta
        self.class_weights = class_weights if class_weights is not None else 1
        self.class_indexes = class_indexes
        self.per_image = per_image
        self.smooth = smooth

    def dice_coef(self, y_true, y_pred, smooth=0):        
        # y_true_f = K.flatten(y_true)
        # y_pred_f = K.flatten(y_pred)
        intersection = K.sum(tf.math.abs(y_true * y_pred))
        dice = (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)
        return dice
    
    def dice(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.math.sigmoid(y_pred)
        numerator = 2 * tf.reduce_sum(y_true * y_pred)
        denominator = tf.reduce_sum(y_true + y_pred)
        return numerator / denominator

    def __call__(self, gt, pr):
        return 1 - self.dice(
            gt,
            pr
        )

class FocalLoss(tf.losses.Loss):

    def __init__(self, alpha = 0.25, gamma = 2,  smooth=1e-5):
        super().__init__(name='focal_loss')
        self.alpha = alpha
        self.gamma = gamma

    def focal_loss_with_logits(self, logits, targets, y_pred):
        targets = tf.cast(targets, tf.float32)
        weight_a = self.alpha * (1 - y_pred) ** self.gamma * targets
        weight_b = (1 - self.alpha) * y_pred ** self.gamma * (1 - targets)
        
        return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b 

    def loss(self, y_true, logits):
        y_pred = tf.math.sigmoid(logits)
        loss_result = self.focal_loss_with_logits(logits=logits, targets=y_true, y_pred=y_pred)

        return tf.reduce_mean(loss_result)

    def focal_loss(self):
        def focal_loss_with_logits(logits, targets, y_pred):
            targets = tf.cast(targets, tf.float32)
            weight_a = self.alpha * (1 - y_pred) ** self.gamma * targets
            weight_b = (1 - self.alpha) * y_pred ** self.gamma * (1 - targets)
            
            return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b 

        def loss(y_true, logits):
            y_pred = tf.math.sigmoid(logits)
            loss_result = focal_loss_with_logits(logits=logits, targets=y_true, y_pred=y_pred)

            return tf.reduce_mean(loss_result)

        return loss

    def __call__(self, gt, pr):
        return self.loss(
            gt,
            pr
        )