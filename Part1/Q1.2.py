# ------------------------------
# Import required libraries
# ------------------------------
import re
import nltk
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import spacy
import string

# Download required NLTK data (only needed once)
nltk.download('punkt')
nltk.download('stopwords')

# ------------------------------
# Define the sample text (Data_1.txt content)
# ------------------------------
text = (
    "Classification is the task of choosing the correct class label for a given input. In basic\n"
    "classification tasks, each input is considered in isolation from all other inputs, and the set of labels is defined in advance. "
    "The basic classification task has a number of interesting variants. For example, in multiclass classification, each instance may be assigned multiple labels; "
    "in open-class classification, the set of labels is not defined in advance; and in sequence classification, a list of inputs are jointly classified."
)

# ------------------------------
# 1.1. Tokenization using Regular Expression
# ------------------------------
regex_tokens = re.findall(r'\b\w+\b', text)
print("Regular Expression Tokenization:")
print(regex_tokens)

# ------------------------------
# 1.2. Tokenization using NLTK
# ------------------------------
nltk_tokens = word_tokenize(text)
print("\nNLTK Tokenization:")
print(nltk_tokens)

# ------------------------------
# 1.3. Tokenization using TextBlob
# ------------------------------
blob = TextBlob(text)
textblob_tokens = blob.words  # Note: blob.words returns a WordList
print("\nTextBlob Tokenization:")
print(list(textblob_tokens))

# ------------------------------
# 1.4. Tokenization using spaCy
# ------------------------------
# Load the small English model (ensure 'en_core_web_sm' is installed)
nlp = spacy.load('en_core_web_sm')
doc = nlp(text)
spacy_tokens = [token.text for token in doc]
print("\nspaCy Tokenization:")
print(spacy_tokens)


# ------------------------------
# Import the stopwords list from NLTK
# ------------------------------
from nltk.corpus import stopwords

# Create a set of English stop words for faster lookup
stop_words = set(stopwords.words('english'))

# Use the tokens obtained via NLTK tokenization (nltk_tokens) for this demonstration
# ------------------------------
# 3.1. Remove stop words and punctuation
filtered_tokens = [
    token for token in nltk_tokens
    if token.lower() not in stop_words and token not in string.punctuation
]

print("\nTokens after removing stop words and punctuation:")
print(filtered_tokens)

# ------------------------------
# 3.2. Identify and print the stop words found in the text corpus
# ------------------------------
stop_words_in_text = [
    token for token in nltk_tokens if token.lower() in stop_words
]
print("\nStop words found in the text corpus:")
print(stop_words_in_text)
