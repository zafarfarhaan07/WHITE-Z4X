
import tensorflow as tf
import numpy as np
import time
import os

from model import Transformer, create_masks
from utils import clean_text, tokenize, save_tokenizer, load_tokenizer, save_config

EN_SENTENCES = ["hello", "how are you", "thank you", "good morning", "i love programming", "good afternoon", "good evening", "good night", "please", "excuse me", "I am sorry", "you are welcome", "nice to meet you", "what is your name?", "my name is John", "how old are you?", "I am twenty years old", "where are you from?", "I am from India", "what do you do?", "I am a student", "do you speak English?", "yes, I speak a little Hindi", "I don't understand", "can you help me?", "where is the bathroom?", "how much does this cost?", "what time is it?", "I am happy", "I am sad", "I am tired", "I am hungry", "I am thirsty", "this is good", "this is bad", "I like this", "I don't like this", "I need help", "I want water", "I want food", "let's go", "stop", "come here", "go there", "I need to go", "see you later", "have a nice day"]
HI_SENTENCES = ["नमस्ते", "आप कैसे हैं", "धन्यवाद", "सुप्रभात", "मुझे प्रोग्रामिंग पसंद है", "शुभ दोपहर", "शुभ संध्या", "शुभ रात्रि", "कृपया", "माफ़ कीजिए", "मुझे माफ़ करना", "आपका स्वागत है", "आपसे मिलकर खुशी हुई", "आपका नाम क्या है?", "मेरा नाम जॉन है", "आपकी उम्र कितनी है?", "मैं बीस साल का हूँ", "आप कहाँ से हैं?", "मैं भारत से हूँ", "आप क्या करते हैं?", "मैं एक छात्र हूँ", "क्या आप अंग्रेजी बोलते हैं?", "हाँ, मैं थोड़ी हिंदी बोलता हूँ", "मुझे समझ नहीं आया", "क्या आप मेरी मदद कर सकते हैं?", "शौचालय कहाँ है?", "यह कितने का है?", "कितने बजे हैं?", "मैं खुश हूँ", "मैं दुखी हूँ", "मैं थका हुआ हूँ", "मुझे भूख लगी है", "मुझे प्यास लगी है", "यह अच्छा है", "यह बुरा है", "मुझे यह पसंद है", "मुझे यह पसंद नहीं है", "मुझे मदद चाहिए", "मुझे पानी चाहिए", "मुझे खाना चाहिए", "चलो चलें", "रुको", "यहाँ आओ", "वहाँ जाओ", "मुझे जाना है", "बाद में मिलते हैं", "आपका दिन शुभ हो"]
HI_SENTENCES = ["<start> " + s + " <end>" for s in HI_SENTENCES]

NUM_LAYERS = 2
D_MODEL = 128
NUM_HEADS = 4
DFF = 512
DROPOUT_RATE = 0.1

EPOCHS = 20
BATCH_SIZE = 2
BUFFER_SIZE = 1000
LEARNING_RATE = 0.001

CHECKPOINT_PATH = "./checkpoints/train"
INP_TOKENIZER_PATH = "./tokenizers/inp_tokenizer.pkl"
TARG_TOKENIZER_PATH = "./tokenizers/targ_tokenizer.pkl"
CONFIG_PATH = "./config/training_config.json"

os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
os.makedirs(os.path.dirname(INP_TOKENIZER_PATH), exist_ok=True)
os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)


print("Starting data preprocessing...")
cleaned_en = [clean_text(s) for s in EN_SENTENCES]
cleaned_hi = [clean_text(s) for s in HI_SENTENCES]


input_tensor, inp_lang_tokenizer = tokenize(cleaned_en, fit=True)
target_tensor, targ_lang_tokenizer = tokenize(cleaned_hi, fit=True)

MAX_LENGTH_INP = input_tensor.shape[1]
MAX_LENGTH_TARG = target_tensor.shape[1]
print(f"Max input sequence length: {MAX_LENGTH_INP}")
print(f"Max target sequence length: {MAX_LENGTH_TARG}")

VOCAB_INP_SIZE = len(inp_lang_tokenizer.word_index) + 1
VOCAB_TAR_SIZE = len(targ_lang_tokenizer.word_index) + 1
print(f"Input vocabulary size: {VOCAB_INP_SIZE}")
print(f"Target vocabulary size: {VOCAB_TAR_SIZE}")

save_tokenizer(inp_lang_tokenizer, INP_TOKENIZER_PATH)
save_tokenizer(targ_lang_tokenizer, TARG_TOKENIZER_PATH)

training_config = {
    "max_length_inp": MAX_LENGTH_INP,
    "max_length_targ": MAX_LENGTH_TARG,
    "vocab_inp_size": VOCAB_INP_SIZE,
    "vocab_tar_size": VOCAB_TAR_SIZE,
    "num_layers": NUM_LAYERS,
    "d_model": D_MODEL,
    "num_heads": NUM_HEADS,
    "dff": DFF,
    "dropout_rate": DROPOUT_RATE
}
save_config(training_config, CONFIG_PATH)


dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor))
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

print("Data preprocessing complete.")

print("Initializing the Transformer model...")
pe_input = MAX_LENGTH_INP + 20
pe_target = MAX_LENGTH_TARG + 20

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
    """Calculates loss, masking padding tokens."""
    mask = tf.math.logical_not(tf.math.equal(real, 0)) 
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask 
    masked_sum = tf.reduce_sum(mask)
    if masked_sum == 0:
         return 0.0 

    return tf.reduce_sum(loss_) / masked_sum


optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.98, epsilon=1e-9
)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(D_MODEL)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_PATH, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    print(f'Restoring from latest checkpoint: {ckpt_manager.latest_checkpoint}')
    dummy_inp = tf.zeros((BATCH_SIZE, MAX_LENGTH_INP), dtype=tf.int64)
    dummy_tar_inp = tf.zeros((BATCH_SIZE, MAX_LENGTH_TARG - 1), dtype=tf.int64)

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(dummy_inp, dummy_tar_inp)
    _ = transformer(dummy_inp, dummy_tar_inp, training=False,
                     enc_padding_mask=enc_padding_mask,
                     look_ahead_mask=combined_mask,
                     dec_padding_mask=dec_padding_mask)
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
    print('Checkpoint successfully restored.')
else:
    print('No checkpoint found, initializing from scratch.')
@tf.function
def train_step(inp, tar):
    """Performs a single training step."""
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp,
                                     training=True,
                                     enc_padding_mask=enc_padding_mask,
                                     look_ahead_mask=combined_mask,
                                     dec_padding_mask=dec_padding_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    return loss

print("Starting training...")
print(f"Training for {EPOCHS} epochs with batch size {BATCH_SIZE}.")
print("WARNING: The dataset is very small. This is for demonstration only and will lead to poor performance.")

train_loss = tf.keras.metrics.Mean(name='train_loss')

for epoch in range(EPOCHS):
    start_time = time.time()
    train_loss.reset_state()

    for (batch, (inp, tar)) in enumerate(dataset):
        batch_loss = train_step(inp, tar)
        train_loss(batch_loss)


    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print(f'Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}')

    epoch_time = time.time() - start_time
    print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Time: {epoch_time:.2f} sec')

print("Training finished.")
print(f"Final checkpoint saved at: {ckpt_manager.latest_checkpoint}")
print(f"Tokenizers saved to: {INP_TOKENIZER_PATH}, {TARG_TOKENIZER_PATH}")
print(f"Config saved to: {CONFIG_PATH}")
