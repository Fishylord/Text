import nltk
import re
from textblob import TextBlob

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

sentence = "The big black dog barked at the white cat and chased away."
# 1. POS Tagging with NLTK
nltk_tokens = nltk.word_tokenize(sentence)
nltk_pos_tags = nltk.pos_tag(nltk_tokens)
print("NLTK POS Tags:", nltk_pos_tags)

# 2. POS Tagging with TextBlob
textblob_obj = TextBlob(sentence)
textblob_pos_tags = textblob_obj.tags
print("TextBlob POS Tags:", textblob_pos_tags)

# 3. POS Tagging with Regular Expressions (Illustrative Example)
regex_patterns = [
    (r'\b(The|the|a|an)\b', 'DET'),
    (r'\b(big|black|white)\b', 'ADJ'),
    (r'\b(dog|cat)\b', 'NN'),
    (r'\b(barked|chased)\b', 'VB'),
    (r'\b(at|and|away)\b', 'IN'),  # Added 'away' to the preposition list
]

regex_pos_tags = []
for word in nltk_tokens:
    for pattern, tag in regex_patterns:
        if re.match(pattern, word):
            regex_pos_tags.append((word, tag))
            break
    else:
        regex_pos_tags.append((word, 'UNK'))

print("Regex POS Tags:", regex_pos_tags)



# 4. Parsing with NLTK
grammar = "NP: {<DET>?<ADJ>*<NN>}"
cp = nltk.RegexpParser(grammar)
result = cp.parse(nltk_pos_tags)
print("Parse Tree:")
print(result)
result.draw() # Uncomment to visualize (requires matplotlib)