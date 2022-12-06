from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras import backend as K
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class resDense(layers.Layer):
    def __init__(self, filters, dropout=0, **kwargs):
        super(resDense, self).__init__(**kwargs)
        self.filters = filters
        self.dropout = dropout
        self.layerA = layers.Dense(filters)
        self.DOA = layers.Dropout(dropout)
        self.batchA = layers.BatchNormalization()
        self.layerB = layers.Dense(filters)
        self.DOB = layers.Dropout(dropout)
        self.batchB = layers.BatchNormalization()

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.filters,
            'dropout': self.dropout
        })
        return config

    def call(self, inputs):
        x = self.layerA(inputs)
        x = self.DOA(x)
        x = self.batchA(x)
        x = self.layerB(x)
        x = self.DOB(x)
        x = self.batchB(x)
        x += inputs
        return x

class CategoricalAttention(layers.Layer):
    def __init__(self, dropout=0.2, **kwargs):
        super(CategoricalAttention, self).__init__(**kwargs)
        self.dropout = dropout
        self.DOA = layers.Dropout(dropout)
        self.batchA = layers.BatchNormalization()

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'dropout': self.dropout
        })
        return config

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[1][1], input_shape[0][1]),
            initializer="random_normal",
            trainable=True,
        )

    def call(self, inputs):
        v = layers.RepeatVector(int(inputs[1].shape[1]))(inputs[0])
        q = layers.Reshape((inputs[1].shape[1],1))(inputs[1])
        #q = inputs[1]
        maskedv = tf.multiply(v, q)
        attmat = tf.multiply(maskedv, self.w)
        resultmat = tf.reduce_sum(attmat, axis=1)
        return resultmat

class CategoricalAttention_v2(layers.Layer):
    def __init__(self, dropout=0.2, **kwargs):
        super(CategoricalAttention_v2, self).__init__(**kwargs)
        self.dropout = dropout
        self.DOA = layers.Dropout(dropout)
        self.batchA = layers.BatchNormalization()

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'dropout': self.dropout
        })
        return config

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[0][1], input_shape[1][1]),
            initializer="random_normal",
            trainable=True,
        )

    def call(self, inputs):
        v = inputs[0]
        q = tf.reduce_sum(self.w, axis=1, keepdims=True)
        attention = tf.math.multiply(q, v)
        return attention

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

class CustomStopper(tf.keras.callbacks.EarlyStopping):
    def __init__(self, monitor='val_accuracy',
                 patience=500, verbose=1, mode='auto',
                 restore_best_weights=True, start_epoch=500, min_acc=0.9):
        super(CustomStopper, self).__init__(monitor=monitor,
                                            patience=patience,
                                            restore_best_weights=restore_best_weights,
                                            verbose=verbose,
                                            mode=mode
                                            )
        self.start_epoch = start_epoch
        self.min_acc = min_acc

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.start_epoch and logs['accuracy']>=self.min_acc:
            super().on_epoch_end(epoch, logs)
