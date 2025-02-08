import collections

def train_bigram_model(corpus):
    """
    Build unigram and bigram counts from the corpus.
    Each sentence in the corpus is assumed to be a string.
    """
    unigram_counts = collections.Counter()
    bigram_counts = collections.Counter()
    
    for sentence in corpus:
        # Tokenize the sentence (split by whitespace)
        tokens = sentence.strip().split()
        unigram_counts.update(tokens)
        # Count bigrams: (w_i, w_i+1)
        for i in range(len(tokens) - 1):
            bigram = (tokens[i], tokens[i+1])
            bigram_counts[bigram] += 1
            
    return unigram_counts, bigram_counts

def sentence_probability(sentence, unigram_counts, bigram_counts, smoothing=False, V=None):
    """
    Compute the probability of a sentence using a bigram model.
    If smoothing is True, apply add-one (Laplace) smoothing.
    V must be provided when smoothing is True.
    """
    tokens = sentence.strip().split()
    prob = 1.0
    for i in range(len(tokens) - 1):
        w1 = tokens[i]
        w2 = tokens[i+1]
        if smoothing:
            # Apply add-one smoothing:
            # (count(w1, w2) + 1) / (count(w1) + V)
            count_bigram = bigram_counts.get((w1, w2), 0)
            count_w1 = unigram_counts.get(w1, 0)
            prob *= (count_bigram + 1) / (count_w1 + V)
        else:
            # Unsmoothed probability:
            count_bigram = bigram_counts.get((w1, w2), 0)
            if count_bigram == 0:
                return 0.0  # If any bigram is unseen, probability is 0.
            count_w1 = unigram_counts[w1]
            prob *= count_bigram / count_w1
    return prob

# -------------------------
# Define the training corpus
training_corpus = [
    "He read a book",
    "I read a different book",
    "He read a book by Danielle"
]

# Build the model
unigram_counts, bigram_counts = train_bigram_model(training_corpus)

# Define the test sentence
test_sentence = "I read a different book by Danielle"

# Calculate unsmoothed probability
unsmoothed_prob = sentence_probability(test_sentence, unigram_counts, bigram_counts, smoothing=False)

# For smoothing, set V as the number of distinct tokens
V = len(unigram_counts)
smoothed_prob = sentence_probability(test_sentence, unigram_counts, bigram_counts, smoothing=True, V=V)

print("Unsmoothed Bigram Probability: {:.6f}".format(unsmoothed_prob))
print("Smoothed Bigram Probability (Add-One): {:.6e}".format(smoothed_prob))
