### 0. Change working directory

```python
ROOT = "/content/drive/MyDrive/CST/SWE_Notebook/Year3Sem1/DAM_202" # Your Working Directory
import os
os.chdir(ROOT)
```

```python
#import os
os.listdir()
```

### 1. Introduction

#### 1.1 Need for data feed

While pretrained models like Google News Word2Vec are powerful, training our own model offers several advantages:

- Domain Specificity: Captures terminology and relationships specific to our field (medical, legal, technical)

- Custom Vocabulary: Includes words and phrases unique to your dataset

- Control: Full control over training parameters and data quality

- Privacy: No need to rely on external models for sensitive data

- Learning: Deep understanding of how Word2Vec actually works

The Neural Network Architecture
Word2Vec uses a simple neural network with three layers:

- Input Layer: One-hot encoded word vectors

- Hidden Layer: Dense representation (the embeddings we want)

- Output Layer: Probability distribution over vocabulary

#### 1.2 CBOW vs Skip-gram Training

- CBOW (Continuous Bag of Words):

  - Input: Context words → Output: Center word

  - Example: ["the", "cat", "on", "mat"] → "sat"

  - Faster training, better for frequent words

  - Good for syntactic relationships

- Skip-gram:

  - Input: Center word → Output: Context words

  - Example: "sat" → ["the", "cat", "on", "mat"]

  - Slower training, better for rare words

  - Excellent for semantic relationships

![Word2Vec Archotectures](Practical_Two\word2vec_architectures.png)

### 2 Training Objectives

The model learns by:

- Maximizing probability of actual word pairs that appear together

- Minimizing probability of random word pairs (negative sampling)

- Adjusting word vectors to achieve these objectives

#### 2.1 Key Training Concepts

**Context Window** Number of words around target word to consider

- Small window (2-3): Captures syntactic relationships

- Large window (5-10): Captures semantic/topical relationships

**Negative Sampling:** Instead of computing probabilities for entire vocabulary, sample a few "negative" examples

- Dramatically speeds up training

- 5-20 negative samples typically used

**Hierarchical Softmax:** Alternative to negative sampling using binary tree structure

- Better for rare words

- More memory efficient for large vocabularies

### 3 Code Implementation

#### 3.1 Data Collection and Preparation

```python
with open('text.txt', 'r', encoding='utf-8') as f: # Remember your data set path should be specified if not in same working directory
    texts = f.readlines()
```

```python
texts[:10]
```

#### 3.2 Data Quality Assessment

```python
def assess_data_quality(texts):
    """Analyze text data quality for Word2Vec training"""

    stats = {
        'total_documents': len(texts),
        'total_words': 0,
        'unique_words': set(),
        'sentence_lengths': [],
        'word_frequencies': {}
    }

    for text in texts:
        words = text.lower().split()
        stats['total_words'] += len(words)
        stats['sentence_lengths'].append(len(words))
        stats['unique_words'].update(words)

        for word in words:
            stats['word_frequencies'][word] = stats['word_frequencies'].get(word, 0) + 1

    stats['vocabulary_size'] = len(stats['unique_words'])
    stats['avg_sentence_length'] = sum(stats['sentence_lengths']) / len(stats['sentence_lengths'])

    # Find most common words
    sorted_words = sorted(stats['word_frequencies'].items(), key=lambda x: x[1], reverse=True)
    stats['top_words'] = sorted_words[:20]

    # Quality indicators
    stats['quality_score'] = {
        'vocabulary_diversity': stats['vocabulary_size'] / stats['total_words'],
        'avg_word_frequency': stats['total_words'] / stats['vocabulary_size'],
        'rare_words_ratio': sum(1 for count in stats['word_frequencies'].values() if count == 1) / stats['vocabulary_size']
    }

    return stats

# Example usage
quality_report = assess_data_quality(texts)
print(f"Total documents: {quality_report['total_documents']:,}")
print(f"Vocabulary size: {quality_report['vocabulary_size']:,}")
print(f"Unique Words: {quality_report['unique_words']}")
print(f"Average sentence length: {quality_report['avg_sentence_length']:.1f}")
print(f"Vocabulary diversity: {quality_report['quality_score']['vocabulary_diversity']:.4f}")
```

#### 3.3 Text Preprocessing Pipeline

```python
#Import Packages
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
import nltk
```

```python
# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')
```

```python
class AdvancedTextPreprocessor:
    """Comprehensive text preprocessing for Word2Vec training"""

    def __init__(self,
                 lowercase=True,
                 remove_punctuation=True,
                 remove_numbers=False,
                 remove_stopwords=False,
                 min_word_length=2,
                 max_word_length=50,
                 lemmatize=False,
                 remove_urls=True,
                 remove_emails=True,
                 keep_sentences=True):

        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        self.min_word_length = min_word_length
        self.max_word_length = max_word_length
        self.lemmatize = lemmatize
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.keep_sentences = keep_sentences

        if remove_stopwords:
            self.stop_words = set(stopwords.words('english'))

        if lemmatize:
            self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        """Clean individual text string"""

        # Remove URLs
        if self.remove_urls:
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove email addresses
        if self.remove_emails:
            text = re.sub(r'\S+@\S+', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        #Combined
         #(r'https?://\S+|www\.\S+|<.*?>|\S+@\S+\.\S+|@\w+|#\w+|[^A-Za-z0-9\s])

        return text

    def tokenize_text(self, text):
        """Tokenize text into sentences or words"""

        if self.keep_sentences:
            # Tokenize into sentences first
            sentences = sent_tokenize(text)
            processed_sentences = []

            for sentence in sentences:
                words = self.process_sentence(sentence)
                if len(words) >= 3:  # Keep sentences with at least 3 words
                    processed_sentences.append(words)

            return processed_sentences
        else:
            # Return single list of words
            return self.process_sentence(text)

    def process_sentence(self, sentence):
        """Process individual sentence"""

        # Lowercase
        if self.lowercase:
            sentence = sentence.lower()

        # Tokenize into words
        words = word_tokenize(sentence)

        processed_words = []
        for word in words:

            # Remove punctuation
            if self.remove_punctuation:
                word = word.translate(str.maketrans('', '', string.punctuation))

            # Skip if empty after punctuation removal
            if not word:
                continue

            # Remove numbers
            if self.remove_numbers and word.isdigit():
                continue

            # Check word length
            if len(word) < self.min_word_length or len(word) > self.max_word_length:
                continue

            # Remove stopwords
            if self.remove_stopwords and word in self.stop_words:
                continue

            # Lemmatize
            if self.lemmatize:
                word = self.lemmatizer.lemmatize(word)

            processed_words.append(word)

        return processed_words

    def preprocess_corpus(self, texts):
        """Preprocess entire corpus"""

        all_sentences = []

        for text in texts:
            if not isinstance(text, str):
                continue

            # Clean text
            cleaned_text = self.clean_text(text)

            # Tokenize and process
            processed = self.tokenize_text(cleaned_text)

            if self.keep_sentences:
                all_sentences.extend(processed)
            else:
                all_sentences.append(processed)

        return all_sentences
```

```python
# Example usage
preprocessor = AdvancedTextPreprocessor(
    lowercase=True,
    remove_punctuation = True,
    remove_numbers=True,
    remove_stopwords=False,  # Keep stopwords for Word2Vec
    lemmatize=False,  # Usually not needed for Word2Vec
    keep_sentences=True
)

# Processing corpus
processed_sentences = preprocessor.preprocess_corpus(texts)
print(f"Processed {len(processed_sentences)} sentences")
print(f"Sample sentence: {processed_sentences[0]}")
```

```python
processed_sentences[:3]
```

#### 3.4 Training Parameters

Parameter Selection Guidelines

```python
def recommend_parameters(corpus_size, vocab_size, domain_type, computing_resources):
    """
    Recommend Word2Vec parameters based on corpus characteristics

    Args:
        corpus_size: Number of sentences/documents
        vocab_size: Unique words in vocabulary
        domain_type: 'general', 'technical', 'social_media', 'academic'
        computing_resources: 'limited', 'moderate', 'high'
    """

    recommendations = {}

    # Vector size based on corpus and vocab size
    if corpus_size < 10000:
        recommendations['vector_size'] = 50
    elif corpus_size < 100000:
        recommendations['vector_size'] = 100
    elif corpus_size < 1000000:
        recommendations['vector_size'] = 200
    else:
        recommendations['vector_size'] = 300

    # Window size based on domain
    domain_windows = {
        'general': 5,
        'technical': 3,  # More syntactic focus
        'social_media': 4,
        'academic': 6    # More semantic focus
    }
    recommendations['window'] = domain_windows.get(domain_type, 5)

    # Min count based on corpus size
    if corpus_size < 10000:
        recommendations['min_count'] = 1
    elif corpus_size < 100000:
        recommendations['min_count'] = 2
    elif corpus_size < 1000000:
        recommendations['min_count'] = 5
    else:
        recommendations['min_count'] = 10

    # Algorithm selection
    if domain_type in ['technical', 'academic']:
        recommendations['sg'] = 1  # Skip-gram for rare technical terms
    else:
        recommendations['sg'] = 0  # CBOW for general text

    # Epochs based on corpus size and resources
    if computing_resources == 'limited':
        recommendations['epochs'] = 5
    elif corpus_size < 100000:
        recommendations['epochs'] = 15
    else:
        recommendations['epochs'] = 10

    # Hierarchical softmax vs negative sampling
    if vocab_size > 100000:
        recommendations['hs'] = 1
        recommendations['negative'] = 0
    else:
        recommendations['hs'] = 0
        recommendations['negative'] = 10

    return recommendations
```

```python
corpus_size = len(processed_sentences)
print(f"Corpus Size: {corpus_size}")

# Calculate vocabulary size (unique words in vocabulary)
vocab = set(word for sentence in processed_sentences for word in sentence)
vocab_size = len(vocab)
print(f"Vocabulary Size: {vocab_size}")
```

```python
# For this task
params = recommend_parameters(
    corpus_size=corpus_size,
    vocab_size=vocab_size,
    domain_type='general',
    computing_resources='moderate'
)
print("Recommended parameters:", params)
```

#### 3.5 Step-by-Step Implementation

Basic Training Implementation

```python
pip install gensim
```

```python
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import time
import multiprocessing

class EpochLogger(CallbackAny2Vec):
    """Callback to log information about training progress"""

    def __init__(self):
        self.epoch = 0
        self.start_time = time.time()

    def on_epoch_begin(self, model):
        print(f"Epoch #{self.epoch} start")

    def on_epoch_end(self, model):
        elapsed = time.time() - self.start_time
        print(f"Epoch #{self.epoch} end - Time elapsed: {elapsed:.2f}s")
        self.epoch += 1

def train_word2vec_model(sentences, save_path=None, **params):
    """
    Train Word2Vec model with given parameters

    Args:
        sentences: List of tokenized sentences
        save_path: Path to save the model
        **params: Word2Vec parameters
    """

    # Set default parameters
    default_params = {
        'vector_size': 100,
        'window': 5,
        'min_count': 5,
        'workers': multiprocessing.cpu_count() - 1,
        'sg': 0,  # CBOW
        'epochs': 10,
        'alpha': 0.025,
        'min_alpha': 0.0001,
        'hs': 0,
        'negative': 10
    }

    # Update with provided parameters
    default_params.update(params)

    print("Training Word2Vec model with parameters:")
    for key, value in default_params.items():
        print(f"  {key}: {value}")

    # Add callback for progress monitoring
    epoch_logger = EpochLogger()

    # Train the model
    print(f"\nTraining on {len(sentences)} sentences...")
    start_time = time.time()

    model = Word2Vec(
        sentences=sentences,
        callbacks=[epoch_logger],
        **default_params
    )

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Vocabulary size: {len(model.wv)} words")

    # Save model if path provided
    if save_path:
        model.save(save_path)
        print(f"Model saved to {save_path}")

    return model
```

```python
# Example usage
model = train_word2vec_model(
    sentences=processed_sentences,
    save_path='my_word2vec_model.model',
    vector_size=50,
    window=2,
    min_count=2,
    epochs=10000,
    compute_loss = True
)
```

```python
vocab_size = len(model.wv.index_to_key)
print("Vocabulary Size:", vocab_size)
```

```python
all_words = model.wv.index_to_key
print("All Words in Vocabulary:", all_words[:10])
```

#### 3.6 Model Evaluation and Validation

Intrinsic Evaluation Methods

```python
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity

class Word2VecEvaluator:
    """Comprehensive evaluation suite for Word2Vec models"""

    def __init__(self, model):
        self.model = model
        self.wv = model.wv

    def evaluate_word_similarity(self, word_pairs_with_scores):
        """
        Evaluate model on word similarity datasets

        Args:
            word_pairs_with_scores: List of tuples (word1, word2, human_score)

        Returns:
            Spearman correlation with human judgments
        """

        model_similarities = []
        human_similarities = []

        for word1, word2, human_score in word_pairs_with_scores:
            try:
                model_sim = self.wv.similarity(word1, word2)
                model_similarities.append(model_sim)
                human_similarities.append(human_score)
            except KeyError:
                # Skip if words not in vocabulary
                continue

        if len(model_similarities) < 10:
            print("Warning: Too few valid word pairs for reliable evaluation")
            return None

        correlation, p_value = spearmanr(human_similarities, model_similarities)

        print(f"Word Similarity Evaluation:")
        print(f"Valid pairs: {len(model_similarities)}")
        print(f"Spearman correlation: {correlation:.4f}")
        print(f"P-value: {p_value:.4f}")

        return correlation

    def evaluate_analogies(self, analogy_dataset):
        """
        Evaluate model on word analogy tasks

        Args:
            analogy_dataset: List of tuples (word_a, word_b, word_c, word_d)
                           representing "word_a is to word_b as word_c is to word_d"

        Returns:
            Accuracy on analogy task
        """

        correct = 0
        total = 0
        #('king', 'queen', 'man', 'woman'),
        for word_a, word_b, word_c, expected_d in analogy_dataset:
            try:
                # Predict word_d
                result = self.wv.most_similar(
                    positive=[word_a, word_b],
                    negative=[word_c],
                    topn=1
                )

                predicted_d = result

                if predicted_d[0][0].lower() == expected_d.lower():
                    correct += 1

                total += 1

            except (KeyError, IndexError):
                # Skip if words not in vocabulary
                continue

        if total == 0:
            print("Warning: No valid analogies found")
            return 0

        accuracy = correct / total

        print(f"Analogy Evaluation:")
        print(f"Valid analogies: {total}")
        print(f"Correct predictions: {correct}")
        print(f"Accuracy: {accuracy:.4f}")

        return accuracy

    def evaluate_odd_one_out(self, word_groups):
        """
        Evaluate model's ability to identify odd words in groups

        Args:
            word_groups: List of lists, each containing words where one doesn't belong

        Returns:
            Accuracy on odd-one-out task
        """

        correct = 0
        total = 0

        for group in word_groups:
            if len(group) < 3:
                continue

            try:
                # Find the word that doesn't match others
                odd_word = self.wv.doesnt_match(group)

                # This is tricky - we need ground truth to evaluate properly
                # For now, just check if the model can identify AN odd word
                correct += 1  # Placeholder - you'd need labeled data
                total += 1

            except KeyError:
                continue

        if total == 0:
            return 0

        accuracy = correct / total

        print(f"Odd-One-Out Evaluation:")
        print(f"  Valid groups: {total}")
        print(f"  Accuracy: {accuracy:.4f}")

        return accuracy

    def analyze_vocabulary_coverage(self, test_texts):
        """
        Analyze how well model vocabulary covers test texts

        Args:
            test_texts: List of text strings

        Returns:
            Coverage statistics
        """

        vocab = set(self.wv.index_to_key)

        total_words = 0
        covered_words = 0
        unknown_words = set()

        for text in test_texts:
            words = text.lower().split()
            total_words += len(words)

            for word in words:
                if word in vocab:
                    covered_words += 1
                else:
                    unknown_words.add(word)

        coverage_ratio = covered_words / total_words if total_words > 0 else 0

        print(f"Vocabulary Coverage Analysis:")
        print(f"  Total words in test: {total_words}")
        print(f"  Covered words: {covered_words}")
        print(f"  Coverage ratio: {coverage_ratio:.4f}")
        print(f"  Unknown words: {len(unknown_words)}")

        return {
            'coverage_ratio': coverage_ratio,
            'unknown_words': list(unknown_words)[:20],  # Show first 20
            'total_unknown': len(unknown_words)
        }

    def compare_with_baseline(self, baseline_model, test_words):
        """
        Compare model performance with baseline model

        Args:
            baseline_model: Another Word2Vec model to compare against
            test_words: List of words to test

        Returns:
            Comparison statistics
        """

        common_words = []
        for word in test_words:
            if word in self.wv and word in baseline_model.wv:
                common_words.append(word)

        if len(common_words) < 10:
            print("Warning: Too few common words for reliable comparison")
            return None

        # Compare similarity patterns
        similarities = []

        for i, word1 in enumerate(common_words[:20]):  # Test subset
            for word2 in common_words[i+1:21]:  # Avoid too many comparisons

                sim1 = self.wv.similarity(word1, word2)
                sim2 = baseline_model.wv.similarity(word1, word2)

                similarities.append((sim1, sim2))

        model_sims = [s for s in similarities]
        baseline_sims = [s for s in similarities]

        correlation, _ = spearmanr(model_sims, baseline_sims)

        print(f"Model Comparison:")
        print(f"  Common vocabulary: {len(common_words)}")
        print(f"  Similarity correlation: {correlation:.4f}")

        return correlation
```

```python
# Example evaluation datasets
word_similarity_pairs = [
    ('king', 'queen', 8.5),
    ('man', 'woman', 8.3),
    ('car', 'automobile', 9.2),
    ('computer', 'laptop', 7.8),
    ('cat', 'dog', 6.1),
    ('happy', 'sad', 2.1),
]

analogy_examples = [
    ('king', 'queen', 'man', 'woman'),
    ('paris', 'france', 'london', 'england'),
    ('walking', 'walked', 'running', 'ran'),
    ('good', 'better', 'bad', 'worse'),
]

# Usage example
evaluator = Word2VecEvaluator(model)
sim_score = evaluator.evaluate_word_similarity(word_similarity_pairs)
analogy_score = evaluator.evaluate_analogies(analogy_examples)
```

Extra

```python
word = "alice"
if word in model.wv:
    similar_words = model.wv.most_similar(word, topn=10)
    print(f"Most similar words to '{word}':")
    for similar_word, similarity in similar_words:
        print(f"{similar_word}: {similarity}")
else:
    print("Word is not in the vocabulary.")
```

```python
model.wv.similarity('king', 'man')
```
