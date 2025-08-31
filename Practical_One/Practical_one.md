# Text Preprocessing: A Comprehensive Guide

This lecture note for practical one covers the fundamental concepts and practical implementation of text preprocessing for Natural Language Processing (NLP). Text preprocessing is a crucial step that transforms raw text into a clean, structured format suitable for machine learning algorithms.

## Overview

Text preprocessing involves several sequential steps that clean and normalize textual data. Raw text often contains inconsistencies, noise, and irrelevant information that can negatively impact model performance. Through systematic preprocessing, we can extract meaningful features and improve the quality of our text analysis.

## Primary Aim:

**"To build a robust, reusable text preprocessing pipeline for downstream NLP tasks"**

## Specific Objectives:

1.  **Text Classification Preparation**: Clean text data for sentiment analysis, spam detection, or document classification
2.  **Information Retrieval**: Prepare documents for search engines or recommendation systems
3.  **Topic Modeling**: Preprocess text corpora for LDA, NMF, or other topic discovery algorithms
4.  **Feature Engineering**: Create meaningful numerical representations from textual data
5.  **Data Quality Improvement**: Remove noise and standardize text format for better model performance

---

## Module Info

### Unit: 3

### Learning Outcome:

- _Implement common text preprocessing techniques_

## Section 0: Creating Data Sets

### Theory Notes

Before diving into preprocessing techniques, we need a sample dataset to work with. In real-world applications, text data comes from various sources like social media posts, customer reviews, documents, or web scraping results.

### Code Implementation

```python

# Import Pandas library

import pandas as pd

```

**Purpose**: Import the essential pandas library for data manipulation and analysis.

```python

data = [

"When life gives you lemons, make lemonade! 🙂",

"She bought 2 lemons for $1 at Maven Market.",

"A dozen lemons will make a gallon of lemonade. [AllRecipes]",

"lemon, lemon, lemons, lemon, lemon, lemons",

"He's running to the market to get a lemon — there's a great sale today.",

"Does Maven Market carry Eureka lemons or Meyer lemons?",

"An Arnold Palmer is half lemonade, half iced tea. [Wikipedia]",

"iced tea is my favorite"

]

```

**Purpose**: Create a sample dataset with diverse text challenges including:

- Mixed case letters

- Punctuation marks

- Special characters and emojis

- Numbers and currency symbols

- Contractions and apostrophes

- Citations in brackets

- Repeated words

```python

# Convert list to DataFrame

data_df = pd.DataFrame(data, columns=['sentence'])

```

**Purpose**: Transform the list into a pandas DataFrame for easier manipulation and analysis.

```python

# Set display options to show full content

pd.set_option('display.max_colwidth', None)

```

**Purpose**: Configure pandas to display complete text content without truncation, essential for examining preprocessing results.

---

## Section 1: Preprocessing

### 1.1 Normalization

### Theory Notes

**Text normalization** is the process of converting text to a standard, consistent format. The most common normalization technique is converting all text to lowercase, which ensures that words like "Apple" and "apple" are treated as the same token.

### Code Implementation

```python

# Create a copy for spaCy processing

spacy_df = data_df.copy()



# Convert text to lowercase

spacy_df['clean_sentence'] = spacy_df['sentence'].str.lower()

```

**Purpose**:

- Create a working copy to preserve original data

- Convert all text to lowercase for consistency

- Store results in a new column called 'clean_sentence'

**Key Learning**: Normalization reduces vocabulary size and prevents case-sensitive duplicates from being treated as different words.

### 1.2 Text Cleaning

### Code Implementation

```python

# Remove specific citations

spacy_df['clean_sentence'] = spacy_df['clean_sentence'].str.replace('[wikipedia]', '')



# Advanced cleaning with regex

combined = r'https?://\S+|www\.\S+|<.*?>|\S+@\S+\.\S+|@\w+|#\w+|[^A-Za-z0-9\s]'

spacy_df['clean_sentence'] = spacy_df['clean_sentence'].str.replace(combined, ' ', regex=True)

spacy_df['clean_sentence'] = spacy_df['clean_sentence'].str.replace(r'\s+', ' ', regex=True).str.strip()

```

**Purpose**:

- Remove citations and references

- Use regular expressions to remove URLs, email addresses, social media handles, and non-alphanumeric characters

- Normalize whitespace by replacing multiple spaces with single spaces

**Regex Pattern Breakdown**:

- `https?://\S+`: Matches HTTP/HTTPS URLs

- `www\.\S+`: Matches www URLs

- `<.*?>`: Matches HTML tags

- `\S+@\S+\.\S+`: Matches email addresses

- `@\w+`: Matches social media mentions

- `#\w+`: Matches hashtags

- `[^A-Za-z0-9\s]`: Matches any character that isn't alphanumeric or whitespace

---

## Section 1.2: Advanced Text Processing with spaCy

### Theory Notes

**spaCy** is a powerful industrial-strength NLP library that provides advanced tokenization, lemmatization, and linguistic analysis. It offers pre-trained language models that understand grammar, syntax, and word relationships.

### Code Implementation

```python

import spacy



# Download and install English language model

!python -m spacy download en_core_web_sm



# Load the pre-trained pipeline

nlp = spacy.load('en_core_web_sm')



# Process a sample sentence

phrase = spacy_df.clean_sentence[0] # "when life gives you lemons make lemonade"

doc = nlp(phrase)

```

**Purpose**:

- Install spaCy's small English model (12.8 MB)

- Load the pre-trained pipeline with tokenization, POS tagging, and lemmatization capabilities

- Create a spaCy document object for processing

### 1.2.1 Tokenization

### Theory Notes

**Tokenization** splits text into individual units (tokens) such as words, punctuation marks, or numbers. Modern tokenizers handle complex cases like contractions, compound words, and special characters intelligently.

### Code Implementation

```python

# Extract tokens as text strings

[token.text for token in doc]

# Output: ['when', 'life', 'gives', 'you', 'lemons', 'make', 'lemonade']



# Extract tokens as spaCy objects (with linguistic attributes)

[token for token in doc]

# Output: [when, life, gives, you, lemons, make, lemonade]

```

**Purpose**:

- Demonstrate two ways to access tokens

- Show how spaCy preserves linguistic information in token objects

### 1.2.2 Lemmatization

### Theory Notes

**Lemmatization** reduces words to their base or root form (lemma) using linguistic knowledge. Unlike stemming, which simply removes suffixes, lemmatization considers the word's part of speech and meaning to find the correct root form.

Examples:

- "running" → "run"

- "better" → "good"

- "mice" → "mouse"

### Code Implementation

```python

# Extract lemmatized forms

[token.lemma_ for token in doc]

# Output: ['when', 'life', 'give', 'you', 'lemon', 'make', 'lemonade']

```

**Purpose**:

- Convert words to their dictionary forms

- Reduce vocabulary size by grouping inflected forms

- Note how "gives" becomes "give" and "lemons" becomes "lemon"

### 1.2.3 Stop Words Removal

### Theory Notes

**Stop words** are common words that carry little semantic meaning and are often filtered out to focus on more meaningful content. Examples include "the", "and", "is", "in", etc.

### Code Implementation

```python

# View all English stop words in spaCy

list(nlp.Defaults.stop_words)

print(f"Total stop words: {len(list(nlp.Defaults.stop_words))}") # 326 stop words



# Remove stop words

[token for token in doc if  not token.is_stop]

# Output: [life, gives, lemons, lemonade]



# Combine lemmatization and stop word removal

[token.lemma_ for token in doc if  not token.is_stop]

# Output: ['life', 'give', 'lemon', 'lemonade']



# Convert back to sentence format

norm = [token.lemma_ for token in doc if  not token.is_stop]

' '.join(norm) # Output: 'life give lemon lemonade'

```

**Purpose**:

- Show the extensive stop word list in spaCy (326 words)

- Demonstrate filtering out common, low-information words

- Combine multiple preprocessing steps for maximum effect

---

## Section 2: Creating Reusable Functions

### Theory Notes

Creating modular, reusable functions is essential for maintainable code and consistent preprocessing across different datasets.

### Code Implementation

```python

# Function for lemmatization and stop word removal

def  token_lemma_stopw(text):

doc = nlp(text)

output = [token.lemma_ for token in doc if  not token.is_stop]

return  ' '.join(output)



# Apply to entire dataset

spacy_df.clean_sentence.apply(token_lemma_stopw)

```

**Purpose**:

- Encapsulate preprocessing logic in a reusable function

- Enable consistent processing across multiple texts

- Demonstrate functional programming approach to text processing

---

## Section 3: Complete NLP Pipeline

### Theory Notes

An **NLP pipeline** combines multiple preprocessing steps into a single, streamlined workflow. This approach ensures consistency and makes it easy to apply the same transformations to new data.

### Code Implementation

```python

def  lower_replace(series):

output = series.str.lower()

combined = r'https?://\S+|www\.\S+|<.*?>|\S+@\S+\.\S+|@\w+|#\w+|[^A-Za-z0-9\s]'

output = output.str.replace(combined, ' ', regex=True)

return output



def  nlp_pipeline(series):

output = lower_replace(series)

output = output.apply(token_lemma_stopw)

return output



# Apply complete pipeline

cleaned_text = nlp_pipeline(data_df.sentence)



# Save processed data for future use

pd.to_pickle(cleaned_text, 'preprocessed_text.pkl')

```

**Purpose**:

- Combine all preprocessing steps into a single function

- Create a reproducible workflow

- Save processed data in pickle format for efficient loading

---

## Section 4: Word Representation (Vectorization)

### Theory Notes

**Vectorization** converts preprocessed text into numerical representations that machine learning algorithms can process. Text must be transformed into vectors (arrays of numbers) because algorithms cannot directly work with text strings.

### 4.1 Count Vectorization (Bag of Words)

### Theory Notes

**Count Vectorization** creates a matrix where each row represents a document and each column represents a unique word in the corpus. Cell values indicate how many times each word appears in each document. This approach ignores word order but captures word frequency.

### Code Implementation

```python

# Load preprocessed data

import pandas as pd

series = pd.read_pickle('preprocessed_text.pkl')



from sklearn.feature_extraction.text import CountVectorizer



# Create Count Vectorizer

cv = CountVectorizer()

bow = cv.fit_transform(series)



# Convert to DataFrame for visualization

pd.DataFrame(bow.toarray(), columns=cv.get_feature_names_out())

```

**Purpose**:

- Transform text into numerical matrix representation

- Each column represents a unique word (feature)

- Each cell contains the count of that word in that document

### Advanced Count Vectorization

```python

# Count Vectorizer with filtering

cv1 = CountVectorizer(

stop_words='english', # Remove English stop words

ngram_range=(1,1), # Use only single words (unigrams)

min_df=2  # Include words that appear in at least 2 documents

)



bow1 = cv1.fit_transform(series)

bow1_df = pd.DataFrame(bow1.toarray(), columns=cv1.get_feature_names_out())



# Calculate term frequencies

term_freq = bow1_df.sum()

```

**Purpose**:

- Apply additional filtering to reduce noise

- Focus on words that appear multiple times across documents

- Calculate overall term frequencies for analysis

---

## Section 5: TF-IDF (Term Frequency-Inverse Document Frequency)

### Theory Notes

**TF-IDF** addresses a key limitation of simple count vectorization by considering both term frequency (how often a word appears in a document) and inverse document frequency (how rare the word is across the entire corpus).

- **Formula**: TF-IDF = TF \times IDF

- **TF (Term Frequency)**: $$\frac {Number \ of \ times  \ word \ appears \ in document} {\ Total \ words \ in \ document}$$

- **IDF (Inverse Document Frequency)**: log $$ \frac ({Total \ documents} {Documents \ containing \ the \ word})$$

**Key Insight**: TF-IDF gives higher weights to words that are frequent in a specific document but rare across the corpus, making them more distinctive and informative.

### Code Implementation

```python

from sklearn.feature_extraction.text import TfidfVectorizer



# Basic TF-IDF vectorization

tv = TfidfVectorizer()

tvidf = tv.fit_transform(series)

tvidf_df = pd.DataFrame(tvidf.toarray(), columns=tv.get_feature_names_out())



# TF-IDF with filtering

tv1 = TfidfVectorizer(min_df=2) # Words must appear in at least 2 documents

tvidf1 = tv1.fit_transform(series)

tvidf1_df = pd.DataFrame(tvidf1.toarray(), columns=tv1.get_feature_names_out())

```

**Purpose**:

- Calculate TF-IDF scores for better feature weighting

- Values closer to 1 indicate highly distinctive words

- Values closer to 0 indicate either common words or absent words

### N-gram Analysis

```python

# Bigram TF-IDF (pairs of consecutive words)

tv2 = TfidfVectorizer(ngram_range=(1,2)) # Include both unigrams and bigrams

tvidf2 = tv2.fit_transform(series)

tvidf2_df = pd.DataFrame(tvidf2.toarray(), columns=tv2.get_feature_names_out())



# Analyze feature importance

tvidf2_df.sum().sort_values(ascending=False)

```

**Purpose**:

- Capture phrase-level information with bigrams

- Examples: "arnold palmer", "buy lemon", "ice tea"

- Preserve some context that unigrams lose

---

## Key Takeaways and Best Practices

### 1. Sequential Processing

Text preprocessing follows a logical sequence:

1.  **Normalization** (lowercase, encoding)

2.  **Cleaning** (remove noise, special characters)

3.  **Tokenization** (split into words)

4.  **Linguistic Processing** (lemmatization, POS tagging)

5.  **Filtering** (stop words, rare words)

6.  **Vectorization** (convert to numerical representation)

### 2. Tool Selection

- **Basic tasks**: Use pandas string methods and regex

- **Advanced linguistics**: Use spaCy for tokenization, lemmatization, and NER

- **Vectorization**: Use scikit-learn's CountVectorizer and TfidfVectorizer

### 3. Parameter Tuning

Critical parameters to consider:

- **min_df**: Minimum document frequency (removes rare words)

- **max_df**: Maximum document frequency (removes very common words)

- **ngram_range**: Include phrases of different lengths

- **stop_words**: Language-specific common words to remove

### 4. Data Pipeline

- Create modular, reusable functions

- Save intermediate results for efficiency

- Maintain original data for reference

- Document parameter choices and rationale

### 5. Quality Control

- Examine results at each preprocessing step

- Visualize term frequencies and distributions

- Validate that preprocessing improves downstream model performance

---

## Practical Applications

This preprocessing pipeline is suitable for:

- **Text Classification** (sentiment analysis, spam detection)

- **Information Retrieval** (search engines, document similarity)

- **Topic Modeling** (LDA, NMF)

- **Named Entity Recognition**

- **Clustering and Recommendation Systems**

The choice of preprocessing steps depends on your specific use case and the nature of your text data. Always validate that preprocessing choices improve your model's performance on the target task.
