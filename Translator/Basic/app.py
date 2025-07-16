
import streamlit as st
import tensorflow as tf
import numpy as np
import time
import os

st.set_page_config(
    page_title="Transformer Translator",
    page_icon=":guardsman:",
    layout="centered",
    initial_sidebar_state="auto"
)

from model import Transformer, create_masks
from utils import clean_text, load_tokenizer, load_config

CHECKPOINT_PATH = "./checkpoints/train"
INP_TOKENIZER_PATH = "./tokenizers/inp_tokenizer.pkl"
TARG_TOKENIZER_PATH = "./tokenizers/targ_tokenizer.pkl"
CONFIG_PATH = "./config/training_config.json"

@st.cache_resource
def load_resources():
    """Loads tokenizers and training configuration."""
    print("Loading resources...")
    inp_tokenizer = load_tokenizer(INP_TOKENIZER_PATH)
    targ_tokenizer = load_tokenizer(TARG_TOKENIZER_PATH)
    config = load_config(CONFIG_PATH)

    if inp_tokenizer is None or targ_tokenizer is None or config is None:
        st.error("Failed to load necessary resources (tokenizers/config). "
                 "Please ensure 'train_transformer.py' has been run successfully "
                 "and the files exist in the correct paths.")
        return None, None, None, None 

    try:
        max_length_inp = config['max_length_inp']
        max_length_targ = config['max_length_targ']
        num_layers = config['num_layers']
        d_model = config['d_model']
        num_heads = config['num_heads']
        dff = config['dff']
        vocab_inp_size = config['vocab_inp_size']
        vocab_tar_size = config['vocab_tar_size']
        dropout_rate = config.get('dropout_rate', 0.1) 
        pe_input = max_length_inp + 20 
        pe_target = max_length_targ + 20 
    except KeyError as e:
        st.error(f"Missing key in configuration file ({CONFIG_PATH}): {e}")
        return None, None, None, None

    print("Resources loaded successfully.")
    return inp_tokenizer, targ_tokenizer, config, (max_length_inp, max_length_targ,
            num_layers, d_model, num_heads, dff, vocab_inp_size, vocab_tar_size,
            dropout_rate, pe_input, pe_target)

inp_lang_tokenizer, targ_lang_tokenizer, training_config, model_params = load_resources()

if model_params:
    (MAX_LENGTH_INP, MAX_LENGTH_TARG, NUM_LAYERS, D_MODEL, NUM_HEADS, DFF,
     VOCAB_INP_SIZE, VOCAB_TAR_SIZE, DROPOUT_RATE, PE_INPUT, PE_TARGET) = model_params
else:
    st.stop()

try:
    START_TOKEN = targ_lang_tokenizer.word_index.get('<start>')
    END_TOKEN = targ_lang_tokenizer.word_index.get('<end>')
    if START_TOKEN is None or END_TOKEN is None:
         st.warning("'<start>' or '<end>' tokens not found in target language vocabulary. "
                    "Translation might not work as expected.")
except AttributeError:
     st.error("Target tokenizer not loaded correctly.")
     START_TOKEN = None
     END_TOKEN = None

@st.cache_resource 
def load_transformer_model(model_params, checkpoint_path):
    """Initializes the Transformer model and loads weights."""
    print("Initializing model structure...")
    (MAX_LENGTH_INP, MAX_LENGTH_TARG, NUM_LAYERS, D_MODEL, NUM_HEADS, DFF,
     VOCAB_INP_SIZE, VOCAB_TAR_SIZE, DROPOUT_RATE, PE_INPUT, PE_TARGET) = model_params

    transformer_model = Transformer(
        num_layers=NUM_LAYERS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dff=DFF,
        input_vocab_size=VOCAB_INP_SIZE,
        target_vocab_size=VOCAB_TAR_SIZE,
        pe_input=PE_INPUT,
        pe_target=PE_TARGET,
        rate=DROPOUT_RATE
    )

    print("Loading model weights...")
    dummy_inp = tf.zeros((1, MAX_LENGTH_INP), dtype=tf.int64)
    dummy_tar_inp = tf.zeros((1, 1), dtype=tf.int64)

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(dummy_inp, dummy_tar_inp)

    try:
        _ = transformer_model(dummy_inp, dummy_tar_inp, training=False,
                              enc_padding_mask=enc_padding_mask,
                              look_ahead_mask=combined_mask,
                              dec_padding_mask=dec_padding_mask)
        print("Model built with dummy data.")
    except Exception as e:
        st.error(f"Error building model with dummy data: {e}")
        return None


    ckpt = tf.train.Checkpoint(transformer=transformer_model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        try:
            ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
            print(f'Latest checkpoint restored from {ckpt_manager.latest_checkpoint}')
            return transformer_model
        except Exception as e:
            st.error(f"Failed to restore model checkpoint: {e}")
            return None
    else:
        st.error(f'No checkpoint found at {checkpoint_path}. Please train the model first.')
        return None

transformer = load_transformer_model(model_params, CHECKPOINT_PATH)

if transformer is None:
    st.stop()


def translate(input_sentence):
    """Translates an input sentence using the trained Transformer model."""
    if START_TOKEN is None or END_TOKEN is None:
         return ["Translation not possible: <start> or <end> tokens missing.", "", {}]
    cleaned_sentence = clean_text(input_sentence)
    input_seq = inp_lang_tokenizer.texts_to_sequences([cleaned_sentence])
    input_seq = tf.keras.preprocessing.sequence.pad_sequences(
        input_seq, maxlen=MAX_LENGTH_INP, padding='post'
    )
    input_seq = tf.constant(input_seq, dtype=tf.int64) 
    output_seq = tf.constant([[START_TOKEN]], dtype=tf.int64)

    translated_tokens = []
    last_attention_weights = {}

    for i in range(MAX_LENGTH_TARG):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(input_seq, output_seq)
        predictions, attention_weights = transformer(
            input_seq,
            output_seq,
            training=False,
            enc_padding_mask=enc_padding_mask,
            look_ahead_mask=combined_mask,
            dec_padding_mask=dec_padding_mask
        )
        predictions = predictions[:, -1:, :]  

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int64) 

        last_attention_weights = attention_weights 

        if tf.equal(predicted_id, END_TOKEN):
            break 

        translated_tokens.append(predicted_id.numpy()[0][0]) 

        output_seq = tf.concat([output_seq, predicted_id], axis=-1)

    translated_sentence = targ_lang_tokenizer.sequences_to_texts([translated_tokens])[0]
    return [cleaned_sentence, translated_sentence, last_attention_weights]

st.title("Transformer English to Hindi/Tamil Translator")

st.write("Enter an English sentence to translate:")

input_sentence = st.text_area("English Sentence", "")

if st.button("Translate"):
    if input_sentence:
        with st.spinner("Translating..."):
            cleaned_input, translated_output, attention_info = translate(input_sentence)

            st.subheader("Original (Cleaned):")
            st.write(cleaned_input)
            st.subheader("Translated:")
            st.write(translated_output)

    else:
        st.warning("Please enter a sentence to translate.")

st.markdown("---")
st.write("Note: This is a demonstration model trained on a very small dataset. "
         "Performance on real-world sentences may be limited.")
