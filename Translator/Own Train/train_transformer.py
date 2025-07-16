import tensorflow as tf
import numpy as np
import time
import os
import re
import unicodedata
import pickle
import json
from datasets import load_dataset



def clean_text(text):
    """
    Cleans the input text by converting to lowercase and removing unwanted characters.
    Keeps English, Hindi (Devanagari), Tamil, spaces, AND <start>/<end> tokens.
    """
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\u0900-\u097F\u0B80-\u0BFF\s<>]", "", text)
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
        for s in sentences:
            s_proc = s
            if not s_proc.startswith('<start>'):
                s_proc = '<start> ' + s_proc
            if not s_proc.endswith(' <end>'):
                s_proc = s_proc + ' <end>'
            processed_sentences.append(s_proc)
    else:
        processed_sentences = sentences

    if fit:
        if tokenizer is None:
            tokenizer_args = {'filters': '', 'oov_token': '<unk>'}
            if vocab_size_limit:
                tokenizer_args['num_words'] = vocab_size_limit
            tokenizer = tf.keras.preprocessing.text.Tokenizer(**tokenizer_args)
        tokenizer.fit_on_texts(processed_sentences)
    elif tokenizer is None:
        raise ValueError("Tokenizer must be provided if fit is False.")

    tensor = tokenizer.texts_to_sequences(processed_sentences if fit else sentences)

    padding_args = {'padding': 'post'}
    if max_len:
        padding_args['maxlen'] = max_len
        padding_args['truncating'] = 'post'

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, **padding_args)

    if fit:
        return tensor, tokenizer
    else:
        return tensor

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


def create_padding_mask(seq):
    """Creates a mask for padding tokens (token id 0)."""
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  

def create_look_ahead_mask(size):
    """Creates a look-ahead mask for the decoder's self-attention."""
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask

def create_masks(inp, tar):
    """Creates all necessary masks for the Transformer."""
    enc_padding_mask = create_padding_mask(inp)
    dec_padding_mask = create_padding_mask(inp) 

    
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask

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
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        """Adds positional encoding to the input tensor."""
        seq_len = tf.shape(inputs)[1]
        slice_len = tf.minimum(tf.shape(self.pos_encoding)[1], seq_len)
        return inputs + self.pos_encoding[:, :slice_len, :]


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights."""
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
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
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask=None):
        """Processes inputs through the multi-head attention mechanism."""
        batch_size = tf.shape(q)[0]
        q = self.wq(q)  
        k = self.wk(k)  
        v = self.wv(v)  

        q = self.split_heads(q, batch_size)  
        k = self.split_heads(k, batch_size)  
        v = self.split_heads(v, batch_size)  
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  
        output = self.dense(concat_attention)  
        return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
    """Creates a two-layer feed-forward network."""
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])

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
        attn_output, _ = self.mha(v=x, k=x, q=x, mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

class DecoderLayer(tf.keras.layers.Layer):
    """Single layer for the Transformer Decoder."""
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads) 
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        """Processes input through the decoder layer."""
        attn1, attn_weights_block1 = self.mha1(v=x, k=x, q=x, mask=look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(v=enc_output, k=enc_output, q=out1, mask=padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2

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
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training=training, mask=mask)

        return x

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

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](
                x, enc_output, training=training,
                look_ahead_mask=look_ahead_mask,
                padding_mask=padding_mask
            )
            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2

        return x, attention_weights

class Transformer(tf.keras.Model):
    """The complete Transformer model."""
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, pe_input, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size) 

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        """Forward pass through the Transformer model."""
        enc_output = self.encoder(inp, training=training, mask=enc_padding_mask)

        dec_output, attention_weights = self.decoder(
            tar, enc_output, training=training,
            look_ahead_mask=look_ahead_mask,
            padding_mask=dec_padding_mask
        )

        final_output = self.final_layer(dec_output)

        return final_output, attention_weights



if __name__ == "__main__":


    MAX_LEN_LIMIT = 100
    VOCAB_SIZE_LIMIT = 20000
    BATCH_SIZE = 2

    NUM_LAYERS = 4
    D_MODEL = 256
    NUM_HEADS = 8
    DFF = 1024
    DROPOUT_RATE = 0.1

    EPOCHS = 20
    BUFFER_SIZE = 1000
    LEARNING_RATE = 0.001

    CHECKPOINT_PATH = "./checkpoints_complete/train"
    INP_TOKENIZER_PATH = "./tokenizers_complete/inp_tokenizer.pkl"
    TARG_TOKENIZER_PATH = "./tokenizers_complete/targ_tokenizer.pkl"
    CONFIG_PATH = "./config_complete/training_config.json"

    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(INP_TOKENIZER_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)

    print("Loading dataset...")
    try:
        ds = load_dataset("Aarif1430/english-to-hindi")
    except Exception as e:
        print(f"Error loading dataset 'Aarif1430/english-to-hindi': {e}")
        print("Please check the dataset name and your internet connection.")
        exit()

    print("Extracting sentences...")
    try:
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

    print("Cleaning and filtering sentences...")
    HI_SENTENCES_CLEANED = []
    EN_SENTENCES_CLEANED = []

    cleaned_hi_temp = [clean_text(s) for s in HI_SENTENCES_RAW]
    cleaned_en_temp = [clean_text(s) for s in EN_SENTENCES_RAW]

    for hi_clean, en_clean in zip(cleaned_hi_temp, cleaned_en_temp):
        if len(hi_clean.split()) <= MAX_LEN_LIMIT and len(en_clean.split()) <= MAX_LEN_LIMIT:
            HI_SENTENCES_CLEANED.append(hi_clean)
            EN_SENTENCES_CLEANED.append(en_clean)

    print(f"Filtered dataset size (max word count {MAX_LEN_LIMIT}): {len(HI_SENTENCES_CLEANED)}")

    if not HI_SENTENCES_CLEANED:
        print("Error: No sentences remained after filtering. Check MAX_LEN_LIMIT or dataset content.")
        exit()

    print("Tokenizing and padding input (English)...")
    input_tensor, inp_lang_tokenizer = tokenize(EN_SENTENCES_CLEANED, fit=True,
                                                max_len=MAX_LEN_LIMIT,
                                                vocab_size_limit=VOCAB_SIZE_LIMIT)

    print("Tokenizing and padding target (Hindi)...")
    target_tensor, targ_lang_tokenizer = tokenize(HI_SENTENCES_CLEANED, fit=True,
                                                  max_len=MAX_LEN_LIMIT,
                                                  vocab_size_limit=VOCAB_SIZE_LIMIT)

    ACTUAL_MAX_LENGTH_INP = input_tensor.shape[1]
    ACTUAL_MAX_LENGTH_TARG = target_tensor.shape[1]
    print(f"Actual input sequence length after padding/truncation: {ACTUAL_MAX_LENGTH_INP}")
    print(f"Actual target sequence length after padding/truncation: {ACTUAL_MAX_LENGTH_TARG}")

    VOCAB_INP_SIZE = len(inp_lang_tokenizer.word_index) + 1
    VOCAB_TAR_SIZE = len(targ_lang_tokenizer.word_index) + 1
    if VOCAB_SIZE_LIMIT:
         VOCAB_INP_SIZE = min(VOCAB_INP_SIZE, VOCAB_SIZE_LIMIT + 1)
         VOCAB_TAR_SIZE = min(VOCAB_TAR_SIZE, VOCAB_SIZE_LIMIT + 1)

    print(f"Input vocabulary size: {VOCAB_INP_SIZE}")
    print(f"Target vocabulary size: {VOCAB_TAR_SIZE}")

    save_tokenizer(inp_lang_tokenizer, INP_TOKENIZER_PATH)
    save_tokenizer(targ_lang_tokenizer, TARG_TOKENIZER_PATH)

    training_config = {
        "max_len_limit": MAX_LEN_LIMIT,
        "vocab_size_limit": VOCAB_SIZE_LIMIT,
        "actual_max_length_inp": ACTUAL_MAX_LENGTH_INP,
        "actual_max_length_targ": ACTUAL_MAX_LENGTH_TARG,
        "vocab_inp_size": VOCAB_INP_SIZE,
        "vocab_tar_size": VOCAB_TAR_SIZE,
        "num_layers": NUM_LAYERS,
        "d_model": D_MODEL,
        "num_heads": NUM_HEADS,
        "dff": DFF,
        "dropout_rate": DROPOUT_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE
    }
    save_config(training_config, CONFIG_PATH)

    print("Creating TensorFlow dataset...")
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor))
    dataset = dataset.shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    print("Data preprocessing complete.")

    print("Initializing the Transformer model...")
    pe_input = ACTUAL_MAX_LENGTH_INP + 20
    pe_target = ACTUAL_MAX_LENGTH_TARG + 20

    transformer = Transformer(
        num_layers=NUM_LAYERS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dff=DFF,
        input_vocab_size=VOCAB_INP_SIZE,
        target_vocab_size=VOCAB_TAR_SIZE,
        pe_input=pe_input,
        pe_target=pe_target,
        rate=DROPOUT_RATE
    )
    print("Model initialized.")

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none'
    )

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_sum(loss_) / (tf.reduce_sum(mask) + 1e-9)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )

    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_PATH, max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print(f'Latest checkpoint restored from {ckpt_manager.latest_checkpoint}')
    else:
        print('No checkpoint found, initializing from scratch.')

    @tf.function
    def train_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions, _ = transformer(
                inp, tar_inp,
                training=True,
                enc_padding_mask=enc_padding_mask,
                look_ahead_mask=combined_mask,
                dec_padding_mask=dec_padding_mask
            )
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        return loss

    print("Starting training...")
    train_loss = tf.keras.metrics.Mean(name='train_loss')

    for epoch in range(EPOCHS):
        start_time = time.time()
        train_loss.reset_state()

        for (batch, (inp, tar)) in enumerate(dataset):
            try:
                batch_loss = train_step(inp, tar)
                train_loss(batch_loss)

                if batch % 100 == 0:
                    print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f}')

            except tf.errors.ResourceExhaustedError as e:
                print(f"\n\n!!! Resource Exhausted (OOM) Error during training step !!!")
                print(f"Epoch: {epoch + 1}, Batch: {batch}")
                print(f"Current Batch Size: {BATCH_SIZE}")
                print("Try reducing BATCH_SIZE, MAX_LEN_LIMIT, or VOCAB_SIZE_LIMIT further.")
                print(f"Error details: {e}")
                exit()
            except Exception as e:
                print(f"\n\n!!! An unexpected error occurred during training step !!!")
                print(f"Epoch: {epoch + 1}, Batch: {batch}")
                print(f"Error details: {e}")
                exit()

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print(f'Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}')

        epoch_time = time.time() - start_time
        print(f'==> Epoch {epoch + 1} Loss {train_loss.result():.4f} Time: {epoch_time:.2f} sec\n')

    ckpt_save_path = ckpt_manager.save()
    print("Training finished.")
    print(f"Final checkpoint saved at: {ckpt_save_path}")
    print(f"Tokenizers saved to: {INP_TOKENIZER_PATH}, {TARG_TOKENIZER_PATH}")
    print(f"Config saved to: {CONFIG_PATH}")
    