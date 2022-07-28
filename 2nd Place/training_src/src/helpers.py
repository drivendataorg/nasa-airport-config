from pathlib import Path
import numpy as np
from tensorflow.keras.utils import to_categorical
import pandas as pd

from src.const import *


def get_file_path(airport, file):
    return Path(DATA_DIR) / airport / f"{airport}_{file}"


def to_cat(y, Num_Classes):

    small_cat = to_categorical(y)
    N, P = small_cat.shape
    cat = np.zeros((N, Num_Classes))

    for p in range(P):
        cat[:, p] = small_cat[:, p]

    return cat


def return_labels(y, labels):

    y_new = []
    for y_val in y:
        y_new.append(labels[y_val])

    y_new = np.array(y_new)

    return y_new
