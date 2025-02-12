# Import necessary libraries
import nltk
print(nltk.data.path)
from nltk import pos_tag, word_tokenize, RegexpTagger, CFG
from textblob import TextBlob

# Download required NLTK data (if not already downloaded)
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

with open("Data_2.txt", "r") as file:
    sentence = file.read().strip()
    print(sentence)

# NLTK POS Tagger
tokens = word_tokenize(sentence)
nltk_tags = pos_tag(tokens)
print("NLTK POS Tagger output:")
print(nltk_tags)

# TextBlob POS Tagger
blob = TextBlob(sentence)
textblob_tags = blob.tags
print("\nTextBlob POS Tagger output:")
print(textblob_tags)

# Regular Expression (Regex) Tagger
# Define regex patterns for common word endings.
patterns = [
    (r'.*ing$', 'VBG'),
    (r'.*ed$', 'VBD'),
    (r'.*es$', 'VBZ'),
    (r'.*ould$', 'MD'),
    (r'.*\'s$', 'NN$'),
    (r'.*s$', 'NNS'),
    (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),
    (r'.*ly$', 'RB'), 
    (r'The', 'DT'),
    (r'.*', 'NN')
]

# Create the regex tagger using the defined patterns.
regexp_tagger = RegexpTagger(patterns)
regex_tags = regexp_tagger.tag(tokens)
print("\nRegular Expression Tagger output:")
print(regex_tags)


# Generating Parse Trees
grammar = CFG.fromstring("""
S    -> NP VP
NP   -> Det AdjP N
AdjP -> Adj | Adj AdjP
VP   -> V PP | V Adv | VP Conj VP
PP   -> P NP
Det  -> 'The' | 'the'
N    -> 'dog' | 'cat'
Adj  -> 'big' | 'black' | 'white'
V    -> 'barked' | 'chased'
Adv -> 'away'
P    -> 'at'
Conj -> 'and'
""")

tokens = ['The', 'big', 'black', 'dog', 'barked', 'at', 'the', 'white', 'cat', 'and', 'chased', 'away']
parser = nltk.ChartParser(grammar)
print("\nParse Trees:")
for tree in parser.parse(tokens):
    print(tree)
    tree.pretty_print()

