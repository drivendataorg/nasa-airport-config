import numpy as np
import pandas as pd
import tensorflow as tf


def gen_loss():

    def binary_loss(y_true, y_pred):

        y_pred = tf.reshape(y_pred, [-1])
        y_true = tf.reshape(y_true, [-1])

        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        return bce(y_true, y_pred)

    return binary_loss

