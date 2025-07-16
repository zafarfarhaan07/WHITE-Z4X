import streamlit as st
import pandas as pd
from datasets import load_dataset
from transformers import pipeline
import re
import string

st.set_page_config(
    page_title="Hindi-English Translator",
    page_icon="🌏",
    layout="centered"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2F80ED;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 500;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .output-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
        color: black; /* Ensure text is visible */
        font-weight: 500;
        white-space: pre-wrap; /* Preserve whitespace and wrap text */
        word-wrap: break-word; /* Break long words */
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        font-size: 0.8rem;
        color: #888;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🇮🇳 Hindi-English Translator 🇬🇧</div>', unsafe_allow_html=True)

HINDI_TO_ENGLISH_MODEL = "Helsinki-NLP/opus-mt-hi-en"
ENGLISH_TO_HINDI_MODEL = "Helsinki-NLP/opus-mt-en-hi"

def preprocess_text(text, is_hindi=False):
    """Basic text cleaning: space out punctuation."""
    if isinstance(text, str): 
        if is_hindi:
            text = re.sub(r'([।|॥])', r' \1 ', text)
        else:
            text = re.sub(r'([.,!?;:])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text).strip()
    return text

@st.cache_resource
def load_hf_dataset():
    """Loads the dataset from Hugging Face and creates dictionaries."""
    try:
        dataset = load_dataset("cfilt/iitb-english-hindi")
        df = pd.DataFrame(dataset['train'])

        if 'translation' in df.columns:
            df['english_sentence'] = df['translation'].apply(lambda x: x.get('en', ''))
            df['hindi_sentence'] = df['translation'].apply(lambda x: x.get('hi', ''))
           
        else:
            st.error("Dataset structure unexpected. Could not find 'translation' column.")
            return pd.DataFrame(), {}, {}

        df['hindi_sentence'] = df['hindi_sentence'].astype(str).fillna('')
        df['english_sentence'] = df['english_sentence'].astype(str).fillna('')

       
        sample_size = min(len(df), 20000) 
        sampled_df = df.sample(n=sample_size, random_state=42)

        hindi_to_english_dict = dict(zip(sampled_df['hindi_sentence'], sampled_df['english_sentence']))
        english_to_hindi_dict = dict(zip(sampled_df['english_sentence'], sampled_df['hindi_sentence']))

        st.success(f"Loaded {len(df)} sentence pairs from Hugging Face dataset for dictionary lookup demonstration.")
        return df, hindi_to_english_dict, english_to_hindi_dict
    except Exception as e:
        st.error(f"Error loading dataset for dictionary lookup: {e}")
        return pd.DataFrame(), {}, {}

@st.cache_resource
def load_hf_translator(model_name):
    """Loads and caches a translation pipeline from Hugging Face."""
    try:
        translator = pipeline("translation", model=model_name, device=-1)
        return translator
    except Exception as e:
        st.error(f"Error loading Hugging Face model '{model_name}': {e}. Please ensure 'transformers' and a backend (like 'torch' or 'tensorflow') are installed and that the model name is correct.")
        return None

def translate_hindi_to_english_dict(hindi_text, translation_dict, df):
    """Translates Hindi to English using dictionary lookup or crude word matching."""
    hindi_text_clean = preprocess_text(hindi_text, is_hindi=True)
    if not hindi_text_clean: return ""

    if hindi_text_clean in translation_dict:
        return translation_dict[hindi_text_clean]
    hindi_words = hindi_text_clean.split()
    word_translation = []
    df_sample = df 

    word_map = {}
    for _, row in df_sample.iterrows():
        hin_sent_words = str(row['hindi_sentence']).split()
        eng_sent_words = str(row['english_sentence']).split()
        for h_word, e_word in zip(hin_sent_words, eng_sent_words):
            h_word_clean = h_word.strip(string.punctuation + '।॥')
            e_word_clean = e_word.strip(string.punctuation + '।॥')
            if h_word_clean and e_word_clean:
                if h_word_clean not in word_map:
                    word_map[h_word_clean] = set()
                word_map[h_word_clean].add(e_word_clean)


    for word in hindi_words:
        word_clean = word.strip(string.punctuation + '।॥')
        if word_clean in word_map:
            translation = list(word_map[word_clean])[0]
            word_translation.append(translation)
        else:
            word_translation.append(f"<{word}>") 

    return " ".join(word_translation) + " (Basic Word Lookup Fallback - Inaccurate)"


def translate_english_to_hindi_dict(english_text, translation_dict, df):
    """Translates English to Hindi using dictionary lookup or crude word matching."""
    english_text_clean = preprocess_text(english_text, is_hindi=False)
    if not english_text_clean: return ""

    if english_text_clean in translation_dict:
        return translation_dict[english_text_clean]

    english_words = english_text_clean.split()
    word_translation = []
    df_sample = df

    word_map = {}
    for _, row in df_sample.iterrows():
        eng_sent_words = str(row['english_sentence']).split()
        hin_sent_words = str(row['hindi_sentence']).split()
        for e_word, h_word in zip(eng_sent_words, hin_sent_words):
            e_word_clean = e_word.strip(string.punctuation)
            h_word_clean = h_word.strip(string.punctuation + '।॥')
            if e_word_clean and h_word_clean:
                if e_word_clean not in word_map:
                    word_map[e_word_clean] = set()
                word_map[e_word_clean].add(h_word_clean)


    for word in english_words:
        word_clean = word.strip(string.punctuation)
        if word_clean in word_map:
            translation = list(word_map[word_clean])[0]
            word_translation.append(translation)
        else:
            word_translation.append(f"<{word}>") 

    return " ".join(word_translation) + " (Basic Word Lookup Fallback - Inaccurate)"



def main():
    with st.spinner("Loading dataset for dictionary lookup demonstration..."):
        df, hindi_to_english_dict, english_to_hindi_dict = load_hf_dataset()

    st.sidebar.markdown('<div class="sub-header">Options</div>', unsafe_allow_html=True)

    translation_direction = st.sidebar.radio(
        label="Select Translation Direction", 
        options=["Hindi → English", "English → Hindi"],
        key="direction",
        label_visibility="visible"
    )

    translation_method = st.sidebar.radio(
        label="Select Translation Method", 
        options=["Neural Model (Higher Accuracy)", "Basic Lookup (Limited Accuracy)"],
        key="method",
        index=0, 
        help="Neural Model uses advanced AI for better translation of general text. Basic Lookup is faster but only works for exact matches or simple words from the dataset and is generally inaccurate.",
        label_visibility="visible" 
    )

    if translation_method == "Basic Lookup (Limited Accuracy)":
        if not df.empty:
            st.sidebar.markdown("---")
            st.sidebar.markdown('<div class="sub-header">Basic Lookup Dataset Info</div>', unsafe_allow_html=True)
            st.sidebar.info(f"Using {len(df)} sentence pairs from the dataset. Note: This method is for demonstration and has limited accuracy.")

            if st.sidebar.checkbox("Show Sample Data (for Basic Lookup)", key="show_basic_lookup_sample", label_visibility="visible"):
                st.sidebar.dataframe(df[['hindi_sentence', 'english_sentence']].head())
        else:
            st.sidebar.warning("Dataset for basic lookup failed to load. This method is unavailable.")

    input_label_text = "Enter Hindi Text" if translation_direction == "Hindi → English" else "Enter English Text"
    placeholder_text = "यहाँ हिंदी में टाइप करें..." if translation_direction == "Hindi → English" else "Type in English here..."
    st.markdown(f'<div class="sub-header">{input_label_text}</div>', unsafe_allow_html=True)
    input_text = st.text_area(label="Input Text Area", height=100, placeholder=placeholder_text, key="input_text", label_visibility="hidden")


    if st.button(label="Translate", key="translate_button"):
        if not input_text.strip():
            st.warning(f"Please enter some text to translate.")
        else:
            output_text = ""
            target_language = "English" if translation_direction == "Hindi → English" else "Hindi"
            method_used_label = ""

            with st.spinner("Translating..."):
                try:
                    if translation_method == "Basic Lookup (Limited Accuracy)":
                        method_used_label = "(Basic Lookup)"
                        if df.empty:
                            st.error("Basic Lookup method unavailable: Dataset failed to load.")
                            output_text = "Error: Dataset not available for Basic Lookup."
                        elif translation_direction == "Hindi → English":
                            output_text = translate_hindi_to_english_dict(input_text, hindi_to_english_dict, df)
                        else: 
                            output_text = translate_english_to_hindi_dict(input_text, english_to_hindi_dict, df)

                    elif translation_method == "Neural Model (Higher Accuracy)":
                        method_used_label = "(Neural Model)"
                        model_name = HINDI_TO_ENGLISH_MODEL if translation_direction == "Hindi → English" else ENGLISH_TO_HINDI_MODEL
                        translator = load_hf_translator(model_name)

                        if translator:
                            results = translator(input_text, max_length=512) 
                            if results and isinstance(results, list) and 'translation_text' in results[0]:
                                output_text = results[0]['translation_text']
                            else:
                                st.error(f"Translation using model {model_name} failed or returned unexpected result.")
                                output_text = "Translation failed."
                        else:
                            output_text = "Error: Neural model could not be loaded."

                    else:
                        output_text = "Invalid translation method selected."

                except Exception as e:
                    st.error(f"An error occurred during translation: {e}")
                    output_text = "Translation Error."

            st.markdown(f'<div class="sub-header">{target_language} Translation {method_used_label}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="output-box">{output_text}</div>', unsafe_allow_html=True)

    st.sidebar.markdown("---")
    with st.sidebar.expander("About This App"):
        st.write(f"""
        This app translates between Hindi and English.

        **Translation Methods:**
        - **Neural Model (Higher Accuracy):** Uses pre-trained `Helsinki-NLP/opus-mt` models (`{HINDI_TO_ENGLISH_MODEL}` and `{ENGLISH_TO_HINDI_MODEL}`) via the `transformers` library. This method leverages advanced AI and is generally recommended for better translation quality of varied text.
        - **Basic Lookup (Limited Accuracy):** Uses the `cfilt/iitb-english-hindi` dataset from Hugging Face ({len(df) if not df.empty else 'N/A'} pairs loaded). It performs exact sentence matches or a very basic word-by-word lookup. This method is primarily for demonstration and is **not accurate** for translating general sentences due to its inability to handle grammar, word order, or context.

        Internet connection is required to download models and dataset on first use.
        """)

    st.markdown("""
    <div class="footer">
        Hindi-English Translator | Uses Streamlit, Hugging Face Datasets & Transformers
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
