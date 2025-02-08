import re
import nltk
from textblob import TextBlob
import spacy
import string

#nltk.download('punkt_tab')
#nltk.download('punkt')
#nltk.download('stopwords')

# --- Read text from file Data_1.txt ---
with open("Data_1.txt", "r", encoding="utf-8") as file:
    text = file.read()

print("=== Original Text (Data_1.txt) ===")
print(text)
print("==================================\n")

# =============================================================================
# 1. Demonstrate word tokenisation using 4 approaches: Regex, NLTK, TextBlob, spaCy
# =============================================================================

# ------------------- 1.1 Regular Expression Tokenization ---------------------
regex_tokens = re.findall(r'\b\w+\b', text)
print("1.1 Regex-based Tokenization:")
print(regex_tokens)
print()

# ------------------- 1.2 NLTK Tokenization -----------------------------------
nltk_tokens = nltk.word_tokenize(text)
print("1.2 NLTK Tokenization:")
print(nltk_tokens)
print()

# ------------------- 1.3 TextBlob Tokenization -------------------------------
blob = TextBlob(text)
textblob_tokens = blob.words  # .words automatically tokenizes
print("1.3 TextBlob Tokenization:")
print(textblob_tokens)
print()

# ------------------- 1.4 spaCy Tokenization ----------------------------------
# Make sure spaCy model is downloaded: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
spacy_tokens = [token.text for token in doc]
print("1.4 spaCy Tokenization:")
print(spacy_tokens)
print()

# =============================================================================
# 2. Justify the most suitable tokenization operation (Discussion in report)
#    - Typically spaCy for advanced NLP tasks
# =============================================================================

# =============================================================================
# 3. Demonstrate stop words & punctuation removal
#    3.1 Using NLTK
# =============================================================================

stop_words = set(nltk.corpus.stopwords.words('english'))
punctuation_list = set(string.punctuation)

# Filtered list: exclude words that are stop words (case-insensitive) or punctuation
filtered_nltk_tokens = [
    word for word in nltk_tokens 
    if word.lower() not in stop_words and word not in punctuation_list
]

print("3.1 NLTK - Stop Words & Punctuation Removed:")
print(filtered_nltk_tokens)
print()

# Identify which stop words were in the text
found_stopwords = {word.lower() for word in nltk_tokens if word.lower() in stop_words}
print("Stop Words Found in Text (via NLTK):")
print(found_stopwords)
print()

# =============================================================================
#    3.2 Using spaCy
# =============================================================================

filtered_spacy_tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
print("3.2 spaCy - Stop Words & Punctuation Removed:")
print(filtered_spacy_tokens)
print()

