# Import necessary libraries
import nltk
print(nltk.data.path)
from nltk import pos_tag, word_tokenize, RegexpTagger, CFG
from textblob import TextBlob

# Download required NLTK data (if not already downloaded)
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# Define the sample sentence
sentence = "The quick brown fox jumps over the lazy dog."

# ---------------------------
# (a) NLTK POS Tagger
# ---------------------------
# Tokenize the sentence and apply NLTK's built-in POS tagger.
tokens = word_tokenize(sentence)
nltk_tags = pos_tag(tokens)
print("NLTK POS Tagger output:")
print(nltk_tags)

# ---------------------------
# (b) TextBlob POS Tagger
# ---------------------------
# Create a TextBlob object and retrieve POS tags.
blob = TextBlob(sentence)
textblob_tags = blob.tags
print("\nTextBlob POS Tagger output:")
print(textblob_tags)

# ---------------------------
# (c) Regular Expression (Regex) Tagger
# ---------------------------
# Define regex patterns for common word endings.
patterns = [
    (r'.*ing$', 'VBG'),       # gerunds
    (r'.*ed$', 'VBD'),        # past tense verbs
    (r'.*es$', 'VBZ'),        # 3rd person singular present
    (r'.*ould$', 'MD'),       # modals
    (r'.*\'s$', 'NN$'),       # possessive nouns
    (r'.*s$', 'NNS'),         # plural nouns
    (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
    (r'.*', 'NN')             # default to noun
]

# Create the regex tagger using the defined patterns.
regexp_tagger = RegexpTagger(patterns)
regex_tags = regexp_tagger.tag(tokens)
print("\nRegular Expression Tagger output:")
print(regex_tags)


# ---------------------------
# (d) Generating Parse Trees
# ---------------------------
# Define a simple grammar for our sentence.
# This grammar supports a noun phrase (NP) with adjectives (AdjP), a verb phrase (VP),
# and a prepositional phrase (PP) for the complement.
grammar = CFG.fromstring("""
S    -> NP VP
NP   -> Det AdjP N
AdjP -> Adj | Adj AdjP
VP   -> V PP
PP   -> P NP
Det  -> 'The' | 'the'
N    -> 'fox' | 'dog'
Adj  -> 'quick' | 'brown' | 'lazy'
V    -> 'jumps'
P    -> 'over'
""")

# Tokenize the sentence (remove punctuation for parsing)
tokens = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']

# Create the chart parser
parser = nltk.ChartParser(grammar)

# Parse the tokens and display all parse trees
print("\nParse Trees:")
for tree in parser.parse(tokens):
    print(tree)
    tree.pretty_print()
