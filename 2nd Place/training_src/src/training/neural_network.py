from email.policy import default
import tensorflow as tf
import autokeras as ak
from tensorflow import keras
from tensorflow import nest
from tensorflow.keras import activations
from tensorflow.keras import layers
from autokeras.blocks import reduction
from autokeras.utils import utils
from keras.utils.generic_utils import get_custom_objects
import keras
import numpy as np
import pandas as pd

from src.const import *


def past_final_layer(airport, lookahead):
    class final_layer(ak.ClassificationHead):
        def build(self, hp, inputs=None):

            features = inputs[1]
            inputs = inputs[0]

            # Get the input_node from inputs.
            inputs = nest.flatten(inputs)
            utils.validate_num_inputs(inputs, 1)
            input_node = inputs[0]
            output_node = input_node

            # Reduce the tensor to a vector.
            if len(output_node.shape) > 2:
                output_node = reduction.SpatialReduction().build(hp, output_node)

            if self.dropout is not None:
                dropout = self.dropout
            else:
                dropout = hp.Choice("dropout", [0.0, 0.25, 0.5], default=0)

            if dropout > 0:
                output_node = layers.Dropout(dropout)(output_node)
            output_node = layers.Dense(self.shape[-1])(output_node)
            if isinstance(self.loss, keras.losses.BinaryCrossentropy):
                output_node = layers.Activation(activations.sigmoid, name=self.name)(
                    output_node
                )
            else:
                output_node = layers.Softmax(name=self.name)(output_node)

            min_support = hp.Choice("min_support", [0.0001], default=0.0001)
            config_support = hp.Float(
                "config_support",
                min_value=0.1,
                max_value=0.97,
                default=0.9 * CONFIG_SUPPORT_DEFAULTS[airport][lookahead],
            )

            ## Miniumum Support and Minimum Config Support ##
            y_pred = output_node
            Num_Class = y_pred.shape[1]
            configs = features[:, -Num_Class:]

            y_pred = y_pred * (1 - config_support)

            # Min Config Support
            y_pred = y_pred + config_support * configs

            # Min Support
            y_pred = y_pred * (1 - min_support * y_pred.shape[1])
            y_pred = y_pred + min_support

            # Return Predictions
            return y_pred

    return final_layer


def gen_loss():
    def binary_loss(y_true, y_pred):

        y_pred = tf.reshape(y_pred, [-1])
        y_true = tf.reshape(y_true, [-1])

        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        return bce(y_true, y_pred)

    return binary_loss


def train_neural_network(X_train, y_train, X_val, y_val, airport, lookahead):

    cbs = [tf.keras.callbacks.EarlyStopping(patience=20)]

    binary_loss = gen_loss()

    get_custom_objects().update({"binary_loss": binary_loss})

    final_layer = past_final_layer(airport, lookahead)

    ## Autokeras Training Routine ##
    input_node = ak.Input()
    output_node = ak.Normalization()(input_node)
    output_node = ak.DenseBlock()(output_node)
    output_node = final_layer()([output_node, input_node])
    clf = ak.AutoModel(
        project_name=f"automodel/{airport}/{lookahead}",
        inputs=input_node,
        outputs=output_node,
        loss=binary_loss,
        overwrite=True,
        max_trials=50,
    )

    # Train Model and Fit Best Hyperparemters
    clf.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        verbose=0,
        callbacks=cbs,
    )

    return clf
