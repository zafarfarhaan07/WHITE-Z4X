import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Attention, Bidirectional
from tensorflow.keras.models import Model
import re
import string

df = pd.read_csv('C:\\Users\\zafar\\Desktop\\Transformer\\hindi-english.csv')

def preprocess_text(text, is_hindi=False):
    if is_hindi:
        text = re.sub(r'([।|॥])', r' \1 ', text)
    else:
        text = re.sub(r'([.,!?;:])', r' \1 ', text)
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    if is_hindi:
        text = 'START ' + text + ' END'
    else:
        text = 'START ' + text + ' END'
    
    return text

df['hindi_processed'] = df['hindi'].apply(lambda x: preprocess_text(x, is_hindi=True))
df['english_processed'] = df['english'].apply(lambda x: preprocess_text(x, is_hindi=False))

hindi_tokenizer = Tokenizer()
english_tokenizer = Tokenizer()

hindi_tokenizer.fit_on_texts(df['hindi_processed'])
english_tokenizer.fit_on_texts(df['english_processed'])

hindi_vocab_size = len(hindi_tokenizer.word_index) + 1
english_vocab_size = len(english_tokenizer.word_index) + 1

hindi_sequences = hindi_tokenizer.texts_to_sequences(df['hindi_processed'])
english_sequences = english_tokenizer.texts_to_sequences(df['english_processed'])

max_hindi_seq_length = max(len(seq) for seq in hindi_sequences)
max_english_seq_length = max(len(seq) for seq in english_sequences)

hindi_padded = pad_sequences(hindi_sequences, maxlen=max_hindi_seq_length, padding='post')
english_padded = pad_sequences(english_sequences, maxlen=max_english_seq_length, padding='post')

encoder_input_data = hindi_padded
decoder_input_data = english_padded[:, :-1]  # Remove the last token (END)
decoder_target_data = english_padded[:, 1:]  # Remove the first token (START)

embedding_dim = 128
lstm_units = 256
dropout_rate = 0.2

encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(hindi_vocab_size, embedding_dim)(encoder_inputs)
encoder_lstm = Bidirectional(LSTM(lstm_units, return_sequences=True, return_state=True, dropout=dropout_rate))
encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_lstm(encoder_embedding)

encoder_h = tf.keras.layers.Concatenate()([forward_h, backward_h])
encoder_c = tf.keras.layers.Concatenate()([forward_c, backward_c])

decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(english_vocab_size, embedding_dim)(decoder_inputs)
decoder_lstm = LSTM(lstm_units * 2, return_sequences=True, return_state=True, dropout=dropout_rate)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=[encoder_h, encoder_c])

attention = tf.keras.layers.Attention()
context_vector = attention([decoder_outputs, encoder_outputs])
decoder_combined_context = tf.keras.layers.Concatenate()([decoder_outputs, context_vector])

decoder_dense = Dense(english_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_combined_context)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(
        [encoder_input_data, decoder_input_data],
        tf.expand_dims(decoder_target_data, -1),
        batch_size=16,
        epochs=50,
        validation_split=0.2
    )

class Translator:
    def __init__(self, model, hindi_tokenizer, english_tokenizer, max_hindi_length, max_english_length):
        self.model = model
        self.hindi_tokenizer = hindi_tokenizer
        self.english_tokenizer = english_tokenizer
        self.max_hindi_length = max_hindi_length
        self.max_english_length = max_english_length
        
        self.encoder_inputs = model.input[0]
        encoder_embedding = model.layers[2].output
        encoder_lstm = model.layers[3]
        encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_lstm(encoder_embedding)
        encoder_h = tf.keras.layers.Concatenate()([forward_h, backward_h])
        encoder_c = tf.keras.layers.Concatenate()([forward_c, backward_c])
        self.encoder_model = Model(self.encoder_inputs, [encoder_outputs, encoder_h, encoder_c])
        
        self.decoder_inputs = Input(shape=(None,))
        self.decoder_state_input_h = Input(shape=(lstm_units * 2,))
        self.decoder_state_input_c = Input(shape=(lstm_units * 2,))
        self.encoder_outputs_input = Input(shape=(max_hindi_length, lstm_units * 2))
        
        decoder_embedding = model.layers[3](self.decoder_inputs)
        decoder_lstm = model.layers[4]
        decoder_outputs, decoder_h, decoder_c = decoder_lstm(
            decoder_embedding, initial_state=[self.decoder_state_input_h, self.decoder_state_input_c]
        )
        
        attention = model.layers[5]
        context_vector = attention([decoder_outputs, self.encoder_outputs_input])
        decoder_combined_context = tf.keras.layers.Concatenate()([decoder_outputs, context_vector])
        
        decoder_dense = model.layers[7]
        decoder_outputs = decoder_dense(decoder_combined_context)
        
        self.decoder_model = Model(
            [self.decoder_inputs, self.encoder_outputs_input, self.decoder_state_input_h, self.decoder_state_input_c],
            [decoder_outputs, decoder_h, decoder_c]
        )
        
        self.english_index_to_word = {v: k for k, v in self.english_tokenizer.word_index.items()}
        self.english_index_to_word[0] = ''  # Padding token
    
    def translate(self, hindi_text):
        hindi_text = preprocess_text(hindi_text, is_hindi=True)
        
        hindi_seq = self.hindi_tokenizer.texts_to_sequences([hindi_text])
        hindi_padded = pad_sequences(hindi_seq, maxlen=self.max_hindi_length, padding='post')
        
        encoder_outputs, encoder_h, encoder_c = self.encoder_model.predict(hindi_padded)
        
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = self.english_tokenizer.word_index.get('START', 1)
        
        decoded_sentence = []
        
        while True:
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq, encoder_outputs, encoder_h, encoder_c]
            )
            
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            
            sampled_word = self.english_index_to_word.get(sampled_token_index, '')
            
            if sampled_word == 'END' or len(decoded_sentence) >= self.max_english_length - 1:
                break
                
            if sampled_word != 'START':
                decoded_sentence.append(sampled_word)
            
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index
            
            encoder_h, encoder_c = h, c
        
        return ' '.join(decoded_sentence)



def translate_hindi_to_english(hindi_text):
    """
    This is a simplified translation function for demonstration.
    For a real application, you would need to train the model.
    """
    translation_dict = dict(zip(df['hindi'], df['english']))
    
    if hindi_text in translation_dict:
        return translation_dict[hindi_text]
    
    hindi_words = hindi_text.split()
    word_translation = []
    
    for word in hindi_words:
        for hindi_sentence, english_sentence in zip(df['hindi'], df['english']):
            if word in hindi_sentence.split():
                hindi_idx = hindi_sentence.split().index(word)
                if hindi_idx < len(english_sentence.split()):
                    word_translation.append(english_sentence.split()[hindi_idx])
                    break
        else:
            word_translation.append(word)
    
    return " ".join(word_translation)

def demo_translation():
    print("Hindi-English Translator Demo")
    print("Enter 'quit' to exit")
    
    while True:
        hindi_text = input("\nEnter Hindi text: ")
        if hindi_text.lower() == 'quit':
            break
        
        english_text = translate_hindi_to_english(hindi_text)
        print(f"English translation: {english_text}")

    demo_translation()