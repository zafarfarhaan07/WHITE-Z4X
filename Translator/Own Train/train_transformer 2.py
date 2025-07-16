import tensorflow as tf
import numpy as np
import time
import os
import re
import unicodedata
import pickle
import json
from datasets import load_dataset


# ==============================================================================
# Utility Functions (from provided utils.py)
# ==============================================================================

def clean_text(text):
    """
    Cleans the input text by converting to lowercase and removing unwanted characters.
    Keeps English, Hindi (Devanagari), Tamil, spaces, AND <start>/<end> tokens.
    """
    text = text.lower()
    # Keep letters (a-z), Devanagari (Hindi), Tamil, spaces, AND <start>/<end> tokens. Remove others.
    # Modified regex to explicitly allow '<start>' and '<end>'
    # Note: Keeping Tamil characters as per the provided utils.py.
    text = re.sub(r"[^a-zA-Z\u0900-\u097F\u0B80-\u0BFF\s<>]", "", text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def tokenize(sentences, tokenizer=None, fit=False, max_len=None, vocab_size_limit=None):
    """
    Tokenizes sentences using TensorFlow Tokenizer.
    Adds <start> and <end> tokens internally if not already present and fit is True.
    Optionally fits the tokenizer on the provided sentences, respecting vocab_size_limit.
    Pads sequences to max_len if provided, otherwise pads to the longest sequence in the batch.
    """
    processed_sentences = []
    if fit:
        # Add start/end tokens only when fitting
        for s in sentences:
            s_proc = s
            if not s_proc.startswith('<start>'):
                s_proc = '<start> ' + s_proc
            if not s_proc.endswith(' <end>'):
                s_proc = s_proc + ' <end>'
            processed_sentences.append(s_proc)
    else:
        # If not fitting, assume sentences already have tokens if needed for prediction
        processed_sentences = sentences

    if fit:
        if tokenizer is None:
            # Create a new tokenizer if none provided and fitting is requested
            tokenizer_args = {'filters': '', 'oov_token': '<unk>'} # filters='' keeps <start>/<end>
            if vocab_size_limit:
                # Keras Tokenizer's num_words includes the OOV token if specified
                tokenizer_args['num_words'] = vocab_size_limit
            tokenizer = tf.keras.preprocessing.text.Tokenizer(**tokenizer_args)
        # Fit the tokenizer on the processed sentences
        tokenizer.fit_on_texts(processed_sentences)
    elif tokenizer is None:
        raise ValueError("Tokenizer must be provided if fit is False.")

    # Convert sentences to sequences of integers
    # Use original sentences if not fitting, processed ones if fitting
    tensor = tokenizer.texts_to_sequences(processed_sentences if fit else sentences)

    # Pad sequences
    padding_args = {'padding': 'post'}
    if max_len:
        padding_args['maxlen'] = max_len
        padding_args['truncating'] = 'post' # Ensure truncation if max_len is set

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, **padding_args)

    # Return tokenizer only if it was created/fitted here
    if fit:
        return tensor, tokenizer
    else:
        return tensor # Only return tensor if tokenizer was passed in

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

# ==============================================================================
# Model Components (from provided model.py)
# ==============================================================================

# --- Masking Functions ---
def create_padding_mask(seq):
    """Creates a mask for padding tokens (token id 0)."""
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # Add extra dimensions to add the padding to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    """Creates a look-ahead mask for the decoder's self-attention."""
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

def create_masks(inp, tar):
    """Creates all necessary masks for the Transformer."""
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp) # Masks based on *encoder* input padding

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar) # Masks based on *decoder* input padding
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask

# --- Positional Encoding ---
class PositionalEncoding(tf.keras.layers.Layer):
    """Injects positional information into the input embeddings."""
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, pos, i, d_model):
        """Calculates the angles for positional encoding."""
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, position, d_model):
        """Generates the positional encoding matrix."""
        angle_rads = self.get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )
        # Apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        # Apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        """Adds positional encoding to the input tensor."""
        seq_len = tf.shape(inputs)[1]
        # Ensure slicing doesn't go out of bounds if seq_len > position
        slice_len = tf.minimum(tf.shape(self.pos_encoding)[1], seq_len)
        return inputs + self.pos_encoding[:, :slice_len, :]


# --- Multi-Head Attention ---
def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights."""
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    # Scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    # Add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9) # Add large negative value where mask is 1
    # Softmax is normalized on the last axis (seq_len_k)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
    return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi-Head Attention layer."""
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3]) # (batch_size, num_heads, seq_len, depth)

    def call(self, v, k, q, mask=None):
        """Processes inputs through the multi-head attention mechanism."""
        batch_size = tf.shape(q)[0]
        q = self.wq(q)  # (batch_size, seq_len_q, d_model)
        k = self.wk(k)  # (batch_size, seq_len_k, d_model)
        v = self.wv(v)  # (batch_size, seq_len_v, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        # scaled_attention: (batch_size, num_heads, seq_len_q, depth)
        # attention_weights: (batch_size, num_heads, seq_len_q, seq_len_k)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        return output, attention_weights

# --- Point-wise Feed Forward Network ---
def point_wise_feed_forward_network(d_model, dff):
    """Creates a two-layer feed-forward network."""
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])

# --- Encoder Layer ---
class EncoderLayer(tf.keras.layers.Layer):
    """Single layer for the Transformer Encoder."""
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        """Processes input through the encoder layer."""
        # Multi-head attention block (self-attention)
        attn_output, _ = self.mha(v=x, k=x, q=x, mask=mask) # Pass mask explicitly
        attn_output = self.dropout1(attn_output, training=training) # Pass training flag
        out1 = self.layernorm1(x + attn_output) # Add & Norm

        # Feed forward network block
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training) # Pass training flag
        out2 = self.layernorm2(out1 + ffn_output) # Add & Norm
        return out2

# --- Decoder Layer ---
class DecoderLayer(tf.keras.layers.Layer):
    """Single layer for the Transformer Decoder."""
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads) # Masked self-attention
        self.mha2 = MultiHeadAttention(d_model, num_heads) # Encoder-decoder attention
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        """Processes input through the decoder layer."""
        # Block 1: Masked Multi-head self-attention
        attn1, attn_weights_block1 = self.mha1(v=x, k=x, q=x, mask=look_ahead_mask) # Pass look_ahead_mask
        attn1 = self.dropout1(attn1, training=training) # Pass training flag
        out1 = self.layernorm1(attn1 + x) # Add & Norm

        # Block 2: Multi-head attention over encoder output
        attn2, attn_weights_block2 = self.mha2(v=enc_output, k=enc_output, q=out1, mask=padding_mask) # Pass padding_mask
        attn2 = self.dropout2(attn2, training=training) # Pass training flag
        out2 = self.layernorm2(attn2 + out1) # Add & Norm

        # Block 3: Point-wise Feed Forward Network
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training) # Pass training flag
        out3 = self.layernorm3(ffn_output + out2) # Add & Norm

        return out3, attn_weights_block1, attn_weights_block2

# --- Encoder ---
class Encoder(tf.keras.layers.Layer):
    """The Encoder part of the Transformer model."""
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(maximum_position_encoding, self.d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        """Processes the input sequence through the encoder stack."""
        seq_len = tf.shape(x)[1]
        # Adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32)) # Scale embedding
        x = self.pos_encoding(x) # Add positional encoding
        x = self.dropout(x, training=training) # Apply dropout

        # Pass through encoder layers
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training=training, mask=mask) # Pass training and mask

        return x  # (batch_size, input_seq_len, d_model)

# --- Decoder ---
class Decoder(tf.keras.layers.Layer):
    """The Decoder part of the Transformer model."""
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(maximum_position_encoding, d_model)
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        """Processes the target sequence through the decoder stack."""
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        # Adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32)) # Scale embedding
        x = self.pos_encoding(x) # Add positional encoding
        x = self.dropout(x, training=training) # Apply dropout

        # Pass through decoder layers
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](
                x, enc_output, training=training, # Pass training flag
                look_ahead_mask=look_ahead_mask, # Pass masks
                padding_mask=padding_mask
            )
            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights

# --- Transformer Model ---
class Transformer(tf.keras.Model):
    """The complete Transformer model."""
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, pe_input, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size) # Output logits

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        """Forward pass through the Transformer model."""
        # Encode the input sequence
        enc_output = self.encoder(inp, training=training, mask=enc_padding_mask) # (batch_size, inp_seq_len, d_model)

        # Decode using the encoded output and the target sequence
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training=training,
            look_ahead_mask=look_ahead_mask,
            padding_mask=dec_padding_mask
        ) # (batch_size, tar_seq_len, d_model)

        # Project decoder output to vocabulary size
        final_output = self.final_layer(dec_output) # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights


# ==============================================================================
# Main Training Script
# ==============================================================================

if __name__ == "__main__":

    # ===============================
    # Configuration & Hyperparameters
    # ===============================

    # --- Memory Optimization Parameters ---
    MAX_LEN_LIMIT = 100      # Limit sequence length during tokenization/padding
    VOCAB_SIZE_LIMIT = 20000 # Limit vocabulary size for tokenizers (Keras num_words)
    BATCH_SIZE = 2         # Reduce batch size (Try 32, 16, 8)

    # --- Model Hyperparameters ---
    NUM_LAYERS = 4           # Number of encoder/decoder layers
    D_MODEL = 256            # Embedding dimension
    NUM_HEADS = 8            # Number of attention heads
    DFF = 1024               # Dimension of feed-forward network
    DROPOUT_RATE = 0.1       # Dropout rate

    # --- Training Parameters ---
    EPOCHS = 20              # Number of training epochs
    BUFFER_SIZE = 1000      # Shuffle buffer size
    LEARNING_RATE = 0.001   # Adam optimizer learning rate

    # --- Paths ---
    # Use different directories for this optimized run to avoid conflicts
    CHECKPOINT_PATH = "./checkpoints_complete/train"
    INP_TOKENIZER_PATH = "./tokenizers_complete/inp_tokenizer.pkl"
    TARG_TOKENIZER_PATH = "./tokenizers_complete/targ_tokenizer.pkl"
    CONFIG_PATH = "./config_complete/training_config.json"

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(INP_TOKENIZER_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)

    # ===============================
    # Data Loading and Preprocessing
    # ===============================
    print("Loading dataset...")
    # Ensure you are using the correct dataset name from Hugging Face Hub
    try:
        ds = load_dataset("Aarif1430/english-to-hindi")
    except Exception as e:
        print(f"Error loading dataset 'Aarif1430/english-to-hindi': {e}")
        print("Please check the dataset name and your internet connection.")
        exit()

    # Extract sentences
    print("Extracting sentences...")
    try:
        # Assuming 'train' split and standard keys
        HI_SENTENCES_RAW = [item['hindi_sentence'] for item in ds['train']]
        EN_SENTENCES_RAW = [item['english_sentence'] for item in ds['train']]
    except KeyError:
        print("Error: Could not find 'hindi_sentence' or 'english_sentence' keys in 'train' split.")
        print("Please inspect the dataset structure (e.g., using ds.column_names) and update the keys.")
        exit()
    except Exception as e:
        print(f"An error occurred during sentence extraction: {e}")
        exit()

    print(f"Original dataset size: {len(HI_SENTENCES_RAW)}")

    # --- Clean and Filter based on Length ---
    print("Cleaning and filtering sentences...")
    HI_SENTENCES_CLEANED = []
    EN_SENTENCES_CLEANED = []

    # Clean sentences first
    cleaned_hi_temp = [clean_text(s) for s in HI_SENTENCES_RAW]
    cleaned_en_temp = [clean_text(s) for s in EN_SENTENCES_RAW]

    # Filter based on word count of *cleaned* text BEFORE adding start/end tokens
    for hi_clean, en_clean in zip(cleaned_hi_temp, cleaned_en_temp):
        # Simple split count for filtering; adjust if needed
        if len(hi_clean.split()) <= MAX_LEN_LIMIT and len(en_clean.split()) <= MAX_LEN_LIMIT:
            # Store the cleaned sentences that pass the filter
            # <start>/<end> tokens will be added by the tokenize function during fitting
            HI_SENTENCES_CLEANED.append(hi_clean)
            EN_SENTENCES_CLEANED.append(en_clean)

    print(f"Filtered dataset size (max word count {MAX_LEN_LIMIT}): {len(HI_SENTENCES_CLEANED)}")

    if not HI_SENTENCES_CLEANED:
        print("Error: No sentences remained after filtering. Check MAX_LEN_LIMIT or dataset content.")
        exit()

    # --- Tokenize and Pad ---
    # The tokenize function now handles adding <start>/<end> when fit=True
    print("Tokenizing and padding input (English)...")
    input_tensor, inp_lang_tokenizer = tokenize(EN_SENTENCES_CLEANED, fit=True,
                                                max_len=MAX_LEN_LIMIT,
                                                vocab_size_limit=VOCAB_SIZE_LIMIT)

    print("Tokenizing and padding target (Hindi)...")
    target_tensor, targ_lang_tokenizer = tokenize(HI_SENTENCES_CLEANED, fit=True,
                                                  max_len=MAX_LEN_LIMIT,
                                                  vocab_size_limit=VOCAB_SIZE_LIMIT)

    # --- Determine Actual Max Lengths AFTER padding/truncation ---
    # The shape[1] will now be exactly MAX_LEN_LIMIT due to padding/truncation
    ACTUAL_MAX_LENGTH_INP = input_tensor.shape[1]
    ACTUAL_MAX_LENGTH_TARG = target_tensor.shape[1]
    print(f"Actual input sequence length after padding/truncation: {ACTUAL_MAX_LENGTH_INP}")
    print(f"Actual target sequence length after padding/truncation: {ACTUAL_MAX_LENGTH_TARG}")

    # --- Vocabulary Sizes ---
    # Keras Tokenizer reserves index 0 for padding.
    # word_index includes OOV token if created. len(word_index) + 1 gives total size.
    # If num_words was set, the effective size is num_words.
    VOCAB_INP_SIZE = len(inp_lang_tokenizer.word_index) + 1
    VOCAB_TAR_SIZE = len(targ_lang_tokenizer.word_index) + 1
    if VOCAB_SIZE_LIMIT:
         # Adjust size based on the limit provided to the tokenizer
         # The tokenizer might have fewer unique words than the limit
         VOCAB_INP_SIZE = min(VOCAB_INP_SIZE, VOCAB_SIZE_LIMIT + 1) # +1 for padding
         VOCAB_TAR_SIZE = min(VOCAB_TAR_SIZE, VOCAB_SIZE_LIMIT + 1) # +1 for padding

    print(f"Input vocabulary size: {VOCAB_INP_SIZE}")
    print(f"Target vocabulary size: {VOCAB_TAR_SIZE}")

    # --- Save Tokenizers and Config ---
    save_tokenizer(inp_lang_tokenizer, INP_TOKENIZER_PATH)
    save_tokenizer(targ_lang_tokenizer, TARG_TOKENIZER_PATH)

    training_config = {
        # Store the limits used during preprocessing
        "max_len_limit": MAX_LEN_LIMIT,
        "vocab_size_limit": VOCAB_SIZE_LIMIT,
        # Store the actual resulting sequence lengths and vocab sizes
        "actual_max_length_inp": ACTUAL_MAX_LENGTH_INP,
        "actual_max_length_targ": ACTUAL_MAX_LENGTH_TARG,
        "vocab_inp_size": VOCAB_INP_SIZE,
        "vocab_tar_size": VOCAB_TAR_SIZE,
        # Store model hyperparameters
        "num_layers": NUM_LAYERS,
        "d_model": D_MODEL,
        "num_heads": NUM_HEADS,
        "dff": DFF,
        "dropout_rate": DROPOUT_RATE,
        # Store training parameters
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE
    }
    save_config(training_config, CONFIG_PATH)

    # --- Create TensorFlow Dataset ---
    print("Creating TensorFlow dataset...")
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor))
    dataset = dataset.shuffle(BUFFER_SIZE)
    # Drop remainder ensures all batches have the same size (BATCH_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    # Prefetch for performance
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    print("Data preprocessing complete.")

    # ===============================
    # Initialize the Model
    # ===============================
    print("Initializing the Transformer model...")
    # Positional encoding needs to cover the max sequence length
    # Use ACTUAL_MAX_LENGTH + buffer just in case, though MAX_LEN_LIMIT should define it
    pe_input = ACTUAL_MAX_LENGTH_INP + 20
    pe_target = ACTUAL_MAX_LENGTH_TARG + 20

    transformer = Transformer(
        num_layers=NUM_LAYERS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dff=DFF,
        input_vocab_size=VOCAB_INP_SIZE,
        target_vocab_size=VOCAB_TAR_SIZE,
        pe_input=pe_input,  # Max pos encoding length for input
        pe_target=pe_target, # Max pos encoding length for target
        rate=DROPOUT_RATE
    )
    print("Model initialized.")

    # ===============================
    # Loss Function and Optimizer
    # ===============================
    # Use sparse categorical crossentropy as targets are integer sequences
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none' # Calculate loss per element, apply mask later
    )

    def loss_function(real, pred):
        # Mask padding tokens (index 0) before calculating mean loss
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask # Apply mask
        # Avoid division by zero if a batch has no non-padding tokens
        return tf.reduce_sum(loss_) / (tf.reduce_sum(mask) + 1e-9)

    # Define the Adam optimizer
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )

    # ===============================
    # Checkpoint Manager
    # ===============================
    # Create checkpoint object to save model and optimizer state
    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
    # Create checkpoint manager to handle saving/restoring checkpoints
    ckpt_manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_PATH, max_to_keep=5)

    # Restore the latest checkpoint if it exists
    if ckpt_manager.latest_checkpoint:
        # Use expect_partial to avoid errors if model/optimizer changed slightly
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print(f'Latest checkpoint restored from {ckpt_manager.latest_checkpoint}')
    else:
        print('No checkpoint found, initializing from scratch.')

    # ===============================
    # Training Step Definition
    # ===============================
    # Wrap the training step in tf.function for graph execution (performance)
    @tf.function
    def train_step(inp, tar):
        # Prepare decoder input and target output
        tar_inp = tar[:, :-1] # Decoder input (remove <end> token)
        tar_real = tar[:, 1:]  # Target output (remove <start> token)

        # Create masks for this batch
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

        # Open a GradientTape to record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # Forward pass through the transformer
            predictions, _ = transformer(
                inp, tar_inp,
                training=True, # Set training mode (enables dropout)
                enc_padding_mask=enc_padding_mask,
                look_ahead_mask=combined_mask,
                dec_padding_mask=dec_padding_mask
            )
            # Calculate the loss
            loss = loss_function(tar_real, predictions)

        # Calculate gradients of the loss w.r.t. model's trainable variables
        gradients = tape.gradient(loss, transformer.trainable_variables)
        # Apply gradients to update the model's weights using the optimizer
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        # Return the loss for this batch
        return loss

    # ===============================
    # Training Loop
    # ===============================
    print("Starting training...")
    # Use Keras metric to track the mean loss per epoch
    train_loss = tf.keras.metrics.Mean(name='train_loss')

    for epoch in range(EPOCHS):
        start_time = time.time()
        # Reset the loss metric at the start of each epoch
        train_loss.reset_state()

        # Iterate over batches in the dataset
        for (batch, (inp, tar)) in enumerate(dataset):
            try:
                # Perform a single training step
                batch_loss = train_step(inp, tar)
                # Update the epoch loss metric
                train_loss(batch_loss)

                # Print progress periodically within the epoch
                if batch % 100 == 0:
                    print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f}')

            # --- Error Handling ---
            except tf.errors.ResourceExhaustedError as e:
                # Specific handling for Out-of-Memory errors
                print(f"\n\n!!! Resource Exhausted (OOM) Error during training step !!!")
                print(f"Epoch: {epoch + 1}, Batch: {batch}")
                print(f"Current Batch Size: {BATCH_SIZE}")
                print("Try reducing BATCH_SIZE, MAX_LEN_LIMIT, or VOCAB_SIZE_LIMIT further.")
                print(f"Error details: {e}")
                exit() # Exit the script on OOM
            except Exception as e:
                # Handle any other unexpected errors during training
                print(f"\n\n!!! An unexpected error occurred during training step !!!")
                print(f"Epoch: {epoch + 1}, Batch: {batch}")
                print(f"Error details: {e}")
                # Consider logging more details or attempting recovery if appropriate
                exit() # Exit on other critical errors

        # --- Save Checkpoint ---
        # Save a checkpoint every N epochs (e.g., every 5 epochs)
        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print(f'Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}')

        # --- Print Epoch Summary ---
        epoch_time = time.time() - start_time
        print(f'==> Epoch {epoch + 1} Loss {train_loss.result():.4f} Time: {epoch_time:.2f} sec\n')

    # --- Final Save ---
    # Save the final checkpoint after training completes
    ckpt_save_path = ckpt_manager.save()
    print("Training finished.")
    print(f"Final checkpoint saved at: {ckpt_save_path}")
    print(f"Tokenizers saved to: {INP_TOKENIZER_PATH}, {TARG_TOKENIZER_PATH}")
    print(f"Config saved to: {CONFIG_PATH}")
    
"""
**Explanation and Key Changes:**

1.  **Integrated Code:** Functions and classes from your `utils.py` and `model.py` are now part of this single script.
2.  **Imports:** Redundant imports for `utils` and `model` are removed.
3.  **Tokenization Logic:** The script now uses the `tokenize` function from your `utils.py`. This function handles adding `<start>` and `<end>` tokens internally when `fit=True`, so the manual addition before calling `tokenize` has been removed from the main script section. It also correctly uses the `max_len` and `vocab_size_limit` parameters.
4.  **Vocabulary Size Calculation:** The calculation for `VOCAB_INP_SIZE` and `VOCAB_TAR_SIZE` is adjusted to correctly account for how the Keras `Tokenizer` handles padding (index 0) and the `num_words` limit.
5.  **Keyword Arguments:** The script consistently uses keyword arguments (e.g., `training=training`, `mask=mask`) when calling layer methods, as implemented in your `model.py`, which is good practice and helps prevent errors.
6.  **Memory Parameters:** The memory-saving parameters (`MAX_LEN_LIMIT`, `VOCAB_SIZE_LIMIT`, `BATCH_SIZE`) are kept at the reduced values.
7.  **Error Handling:** Basic `try...except` blocks are included in the training loop to catch potential `ResourceExhaustedError` (OOM) and other exceptions, providing more informative messages.
8.  **Clarity and Comments:** Added comments to explain different sections and key steps.

**To Run:**

1.  **Save:** Save the code above as a Python file (e.g., `train_transformer_complete.py`).
2.  **Install Libraries:** Make sure you have the necessary libraries installed:
    ```bash
    pip install tensorflow datasets numpy huggingface_hub
    ```
3.  **Execute:** Run the script from your terminal:
    ```bash
    python train_transformer_complete.py
    ```

Monitor the output and your system's RAM usage. If you still encounter OOM errors, the primary parameter to reduce further is `BATCH_SIZE`. You might also need to decrease `MAX_LEN_LIMIT` or `VOCAB_SIZE_LIMIT` if memory constraints are very tig"""