import tensorflow as tf
import numpy as np
import pandas as pd
from model.utils import load_data, tf_lower_and_split_punct, tf_lower_and_split_punct_hindi, masked_acc, masked_loss
from model.modules.encoder import Encoder
from model.modules.cross_attention import CrossAttention
from model.modules.decoder import Decoder
from model.model import Translator

# Load training dataframe
translate_df = pd.read_csv(
    'dataset\hindi-english-parallel-corpus\hindi_english_parallel.csv')
print(translate_df.head())

context_raw, target_raw = load_data(translate_df, max_len=200000)

BUFFER_SIZE = len(context_raw)
BATCH_SIZE = 64

is_train = np.random.uniform(size=(len(target_raw),)) < 0.8

train_raw = (
    tf.data.Dataset
    .from_tensor_slices((context_raw[is_train], target_raw[is_train]))
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE))
val_raw = (
    tf.data.Dataset
    .from_tensor_slices((context_raw[~is_train], target_raw[~is_train]))
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE))

del context_raw, target_raw

max_vocab_size = 50000

context_text_processor = tf.keras.layers.TextVectorization(
    standardize=tf_lower_and_split_punct,
    max_tokens=max_vocab_size,
    ragged=True)

context_text_processor.adapt(train_raw.map(lambda context, target: context))

target_text_processor = tf.keras.layers.TextVectorization(
    standardize=tf_lower_and_split_punct_hindi,
    max_tokens=max_vocab_size,
    ragged=True)
context_text_processor.get_vocabulary()[:20]

target_text_processor.adapt(train_raw.map(lambda context, target: target))
target_text_processor.get_vocabulary()[:20]


def process_text(context, target):
    context = context_text_processor(context).to_tensor()
    target = target_text_processor(target)
    targ_in = target[:, :-1].to_tensor()
    targ_out = target[:, 1:].to_tensor()
    return (context, targ_in), targ_out


train_ds = train_raw.map(process_text, tf.data.AUTOTUNE)
val_ds = val_raw.map(process_text, tf.data.AUTOTUNE)

UNITS = 256


# Encode the input sequence.
encoder = Encoder(context_text_processor, UNITS)

# Attend to the encoded tokens
attention_layer = CrossAttention(UNITS)
embed = tf.keras.layers.Embedding(
    target_text_processor.vocabulary_size(), output_dim=UNITS, mask_zero=True)

# Decoder
decoder = Decoder(target_text_processor, UNITS)


# Model Initialization
model = Translator(UNITS, context_text_processor, target_text_processor)

model.compile(optimizer='adam', loss=masked_loss,
              metrics=[masked_acc, masked_loss])

vocab_size = 1.0 * target_text_processor.vocabulary_size()

model.evaluate(val_ds, steps=20, return_dict=True)

history = model.fit(
    train_ds.repeat(),
    epochs=100,
    steps_per_epoch=100,
    validation_data=val_ds,
    validation_steps=20)

inputs = [
    "It's really cold here.",
    "This is my life.",
    "His room is a mess"
]


for t in inputs:
    print(model.translate([t])[0].numpy().decode())

print()
