import tensorflow as tf
import tensorflow_text as tf_text
import einops
import numpy as np
import re
from tqdm import tqdm


class ShapeChecker():
    def __init__(self):
        # Keep a cache of every axis-name seen
        self.shapes = {}

    def __call__(self, tensor, names, broadcast=False):
        if not tf.executing_eagerly():
            return

        parsed = einops.parse_shape(tensor, names)

        for name, new_dim in parsed.items():
            old_dim = self.shapes.get(name, None)

            if (broadcast and new_dim == 1):
                continue

            if old_dim is None:
                # If the axis name is new, add its length to the cache.
                self.shapes[name] = new_dim
                continue

            if new_dim != old_dim:
                raise ValueError(f"Shape mismatch for dimension: '{name}'\n"
                                 f"    found: {new_dim}\n"
                                 f"    expected: {old_dim}\n")


# Lowercase, trim, and remove non-letter characters
def normalizeEng(s):
    s = str(s).lower().strip()
    s = re.sub(r"([.!?|,])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()


def normalizeHindi(s):
    s = str(s).lower().strip()
    s = re.sub(r"([.!?|,])", r" \1", s)
    s = re.sub(r"([\u0964])", r" \1", s)
    return s.strip()


def load_data(dataframe, max_len=None):
    if max_len is None:
        max_len = len(dataframe)
    if max_len is not None and max_len > len(dataframe):
        max_len = len(dataframe)

    cols = dataframe.columns.tolist()
    context = []
    target = []
    for idx in tqdm(range(max_len)):
        context.append(normalizeHindi(dataframe[cols[0]].loc[idx]))
        target.append(normalizeEng(dataframe[cols[1]].loc[idx]))

    context = np.array(context)
    target = np.array(target)

    return target, context


def tf_lower_and_split_punct(text):
    # Split accented characters.
    text = tf_text.normalize_utf8(text, 'NFKD')
    text = tf.strings.lower(text)
    # Keep space, a to z, and select punctuation.
    text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')
    # Add spaces around punctuation.
    text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
    # Strip whitespace.
    text = tf.strings.strip(text)

    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    return text


def tf_lower_and_split_punct_hindi(text):
    # Add Start and End
    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    return text


def masked_loss(y_true, y_pred):
    # Calculate the loss for each item in the batch.
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    loss = loss_fn(y_true, y_pred)

    # Mask off the losses on padding.
    mask = tf.cast(y_true != 0, loss.dtype)
    loss *= mask

    # Return the total.
    return tf.reduce_sum(loss)/tf.reduce_sum(mask)


def masked_acc(y_true, y_pred):
    # Calculate the loss for each item in the batch.
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, y_true.dtype)

    match = tf.cast(y_true == y_pred, tf.float32)
    mask = tf.cast(y_true != 0, tf.float32)

    return tf.reduce_sum(match)/tf.reduce_sum(mask)
