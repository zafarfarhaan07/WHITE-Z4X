
import tensorflow as tf
import numpy as np
import re
import json
import pickle
import os

def clean_text(text):
    """
    Cleans the input text by converting to lowercase and removing unwanted characters.
    Keeps English, Hindi (Devanagari), Tamil, spaces, AND <start>/<end> tokens.
    """
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\u0900-\u097F\u0B80-\u0BFF\s<>]", "", text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def tokenize(sentences, tokenizer=None, fit=False, max_len=None):
    """
    Tokenizes sentences using TensorFlow Tokenizer.
    Adds <start> and <end> tokens internally if not already present and fit is True.
    Optionally fits the tokenizer on the provided sentences.
    Pads sequences to max_len if provided, otherwise pads to the longest sequence in the batch.
    """
    processed_sentences = []
    if fit:
        for s in sentences:
            if not s.startswith('<start>'):
                 s = '<start> ' + s
            if not s.endswith(' <end>'):
                 s = s + ' <end>'
            processed_sentences.append(s)
    else:
        processed_sentences = sentences


    if fit:
        if tokenizer is None:
            tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<unk>')
        tokenizer.fit_on_texts(processed_sentences)
    elif tokenizer is None:
        raise ValueError("Tokenizer must be provided if fit is False.")

    tensor = tokenizer.texts_to_sequences(processed_sentences if fit else sentences)


    if max_len:
        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, maxlen=max_len, padding='post')
    else:
        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

    return tensor, tokenizer

def save_tokenizer(tokenizer, filepath):
    """Saves a Keras Tokenizer to a file using pickle."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    try:
        with open(filepath, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Tokenizer saved to {filepath}")
    except Exception as e:
        print(f"Error saving tokenizer to {filepath}: {e}")


def load_tokenizer(filepath):
    """Loads a Keras Tokenizer from a file using pickle."""
    try:
        with open(filepath, 'rb') as handle:
            tokenizer = pickle.load(handle)
        print(f"Tokenizer loaded from {filepath}")
        return tokenizer
    except FileNotFoundError:
        print(f"Error: Tokenizer file not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading tokenizer from {filepath}: {e}")
        return None

def save_config(config, filepath):
    """Saves configuration dictionary to a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    try:
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Configuration saved to {filepath}")
    except Exception as e:
        print(f"Error saving configuration to {filepath}: {e}")

def load_config(filepath):
    """Loads configuration dictionary from a JSON file."""
    try:
        with open(filepath, 'r') as f:
            config = json.load(f)
        print(f"Configuration loaded from {filepath}")
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading configuration from {filepath}: {e}")
        return None
