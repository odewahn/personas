"""
Stylometric Profiler
-------------------
This script analyzes a corpus of text from an expert to generate a comprehensive
stylometric profile for use in LLM persona creation.

The analysis includes:
- Lexical features (vocabulary, word frequencies, n-grams)
- Syntactic patterns (sentence structure, complexity)
- Readability metrics
- Discourse markers and pragmatic elements
- Topic modeling
- LIWC-like analysis of psychological dimensions
- Extraction of representative examples

This profile can be used to create prompt templates that guide LLMs to respond
in the style of the analyzed expert.

Usage:
    python stylometric_profiler.py --corpus_dir "./expert_corpus" --output "expert_profile"

Requirements:
    Python 3.7-3.11 (Gensim is not compatible with Python 3.13)
    pip install spacy nltk pandas numpy matplotlib seaborn textstat scikit-learn gensim
    
    # Install spaCy models manually before running:
    python -m spacy download en_core_web_sm  # Smaller model, or use en_core_web_md/lg for better results
    
    # NLTK data will be downloaded automatically when the script runs
    
Note: If spaCy models are not installed, the script will fall back to basic NLP processing
with reduced accuracy for syntactic analysis.
"""

import os
import sys
import json
import re
import argparse
import subprocess
from collections import Counter
from typing import List, Dict, Any, Tuple

# Data processing and analysis
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# NLP libraries
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
import textstat

# Topic modeling
from gensim.models import LdaModel
from gensim.corpora import Dictionary

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Custom sentence tokenizer to avoid NLTK's punkt_tab issue
def custom_sent_tokenize(text):
    """
    A simple sentence tokenizer that doesn't rely on NLTK's punkt_tab.
    This handles common sentence endings and preserves paragraph structure.
    """
    # Handle common sentence endings (., !, ?)
    # This regex looks for sentence endings followed by space and uppercase letter
    # or sentence endings at the end of the text
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$', text)
    
    # Further split by newlines that likely indicate paragraph breaks
    result = []
    for sentence in sentences:
        # Split by double newlines (paragraphs)
        parts = re.split(r'\n\s*\n', sentence)
        for part in parts:
            # Split by single newlines that are followed by uppercase letters (likely new sentences)
            subparts = re.split(r'\n(?=[A-Z])', part)
            result.extend(subparts)
    
    # Clean up the results
    return [s.strip() for s in result if s.strip()]

# Download required resources if not already available
def download_nltk_data():
    """Download required NLTK data packages if they're not already available."""
    resources = [
        ('stopwords', 'corpora/stopwords')
    ]
    
    for resource, path in resources:
        try:
            nltk.data.find(path)
            print(f"NLTK {resource} already downloaded.")
        except LookupError:
            print(f"Downloading NLTK {resource}...")
            nltk.download(resource, quiet=False)
            print(f"Downloaded NLTK {resource}.")

# Download required NLTK data
download_nltk_data()


def preprocess_corpus(corpus_files: List[str]) -> Dict[str, Any]:
    """
    Load and preprocess text from a collection of documents.

    Args:
        corpus_files: List of file paths to process

    Returns:
        Dictionary containing processed corpus data:
            - documents: List of full document texts
            - sentences: List of all sentences
            - nlp_docs: List of spaCy processed documents
    """
    print(f"Processing {len(corpus_files)} documents...")

    # Load spaCy model for linguistic analysis
    spacy_available = True
    try:
        nlp = spacy.load("en_core_web_lg")
    except OSError:
        # Try to load a smaller model if the large one isn't available
        try:
            print("Large model not found. Trying to load medium-sized model...")
            nlp = spacy.load("en_core_web_md")
        except OSError:
            try:
                print("Medium model not found. Trying to load small model...")
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("No spaCy models found. Using basic NLP processing instead.")
                print("To use advanced features, please install a spaCy model manually with:")
                print("python -m spacy download en_core_web_sm")
                spacy_available = False
                # Create a minimal replacement for spaCy's nlp
                class SimpleDoc:
                    def __init__(self, text):
                        self.text = text
                        # Use our custom sentence tokenizer instead of NLTK's
                        self.sents = [SimpleSpan(s) for s in custom_sent_tokenize(text)]
                
                class SimpleSpan:
                    def __init__(self, text):
                        self.text = text
                        words = word_tokenize(text)
                        self.tokens = [SimpleToken(w) for w in words]
                    
                    def __len__(self):
                        return len(self.tokens)
                    
                    def __iter__(self):
                        return iter(self.tokens)
                
                class SimpleToken:
                    def __init__(self, text):
                        self.text = text
                        self.pos_ = "NOUN" if text[0].isupper() else "VERB" if text.endswith(('ing', 'ed')) else "ADJ" if text.endswith(('ly')) else "DET" if text.lower() in ('a', 'an', 'the') else "ADP" if text.lower() in ('in', 'on', 'at', 'by', 'for') else "NOUN"
                        self.dep_ = "nsubj" if text[0].isupper() else "ROOT" if self.pos_ == "VERB" else "dobj" if self.pos_ == "NOUN" else "det" if self.pos_ == "DET" else "prep" if self.pos_ == "ADP" else "amod"
                
                def simple_nlp(text):
                    return SimpleDoc(text)
                
                nlp = simple_nlp

    # Initialize storage structures
    documents = []
    sentences = []

    # Process each document
    for file_path in corpus_files:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()

                # Store the full document
                documents.append(text)

                # Split into sentences and store using our custom tokenizer
                doc_sentences = custom_sent_tokenize(text)
                sentences.extend(doc_sentences)
        except UnicodeDecodeError:
            # Try with different encoding if UTF-8 fails
            try:
                with open(file_path, "r", encoding="latin-1") as file:
                    text = file.read()
                    documents.append(text)
                    doc_sentences = custom_sent_tokenize(text)
                    sentences.extend(doc_sentences)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue

    print(f"Processed {len(documents)} documents with {len(sentences)} total sentences")

    # Process documents with spaCy or fallback (for linguistic analysis)
    print("Performing linguistic analysis...")
    nlp_docs = []
    for doc in documents:
        # Process each document, limiting length if needed
        if len(doc) > 1000000:  # If document is very large
            chunks = [doc[i : i + 1000000] for i in range(0, len(doc), 1000000)]
            for chunk in chunks:
                nlp_docs.append(nlp(chunk))
        else:
            nlp_docs.append(nlp(doc))

    return {"documents": documents, "sentences": sentences, "nlp_docs": nlp_docs}


def extract_lexical_features(corpus_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract lexical features like vocabulary richness, word frequencies, etc.

    Args:
        corpus_data: Dictionary containing the processed corpus

    Returns:
        Dictionary of lexical features including:
            - vocabulary_size: Number of unique words
            - type_token_ratio: Measure of lexical diversity
            - common_words: Most frequent content words
            - function_word_dist: Function word frequencies
            - common_bigrams/trigrams: Common word sequences
    """
    print("Extracting lexical features...")

    # Combine all text for word-level analysis
    all_text = " ".join(corpus_data["documents"])
    words = word_tokenize(all_text.lower())

    # Remove stopwords for content word analysis
    stop_words = set(stopwords.words("english"))
    content_words = [w for w in words if w.isalpha() and w not in stop_words]

    # Calculate vocabulary richness
    unique_words = set(content_words)
    total_words = len(content_words)

    # Type-token ratio (measure of lexical diversity)
    ttr = len(unique_words) / total_words if total_words > 0 else 0

    # Word frequency distribution
    word_freq = Counter(content_words)

    # Function word analysis
    # Expanded list of English function words
    function_words = [
        "the",
        "of",
        "and",
        "a",
        "to",
        "in",
        "is",
        "you",
        "that",
        "it",
        "he",
        "was",
        "for",
        "on",
        "are",
        "as",
        "with",
        "his",
        "they",
        "I",
        "at",
        "be",
        "this",
        "have",
        "from",
        "or",
        "one",
        "had",
        "by",
        "word",
        "but",
        "not",
        "what",
        "all",
        "were",
        "we",
        "when",
        "your",
        "can",
        "said",
        "there",
        "use",
        "an",
        "each",
        "which",
        "she",
        "do",
        "how",
        "their",
        "if",
        "will",
        "up",
        "other",
        "about",
        "out",
        "many",
        "then",
        "them",
        "these",
        "so",
        "some",
        "her",
        "would",
        "make",
        "like",
        "him",
        "into",
        "time",
        "has",
        "look",
    ]

    total_words_count = len(words)
    function_word_counts = {
        word: words.count(word) / total_words_count for word in function_words
    }

    # N-gram analysis (for common phrases)
    bigrams = list(ngrams(words, 2))
    trigrams = list(ngrams(words, 3))
    common_bigrams = Counter(bigrams).most_common(50)
    common_trigrams = Counter(trigrams).most_common(50)

    # Calculate average word length
    word_lengths = [len(word) for word in content_words if word.isalpha()]
    avg_word_length = np.mean(word_lengths) if word_lengths else 0

    return {
        "vocabulary_size": len(unique_words),
        "type_token_ratio": ttr,
        "average_word_length": avg_word_length,
        "common_words": word_freq.most_common(100),
        "function_word_dist": function_word_counts,
        "common_bigrams": common_bigrams,
        "common_trigrams": common_trigrams,
    }


def analyze_syntactic_patterns(corpus_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze sentence structures, complexity, and grammatical patterns.

    Args:
        corpus_data: Dictionary containing the processed corpus

    Returns:
        Dictionary of syntactic features including:
            - sentence length statistics
            - POS patterns
            - dependency patterns
            - voice usage (active vs passive)
    """
    print("Analyzing syntactic patterns...")

    nlp_docs = corpus_data["nlp_docs"]
    sentences = corpus_data["sentences"]

    # Sentence length distribution
    sentence_lengths = [len(sent.split()) for sent in sentences]

    # Parse trees and dependency patterns
    dependency_patterns = []
    pos_patterns = []
    clause_counts = []

    try:
        for doc in nlp_docs:
            # Analyze each sentence
            for sent in doc.sents:
                # Skip very short sentences as they might be headers or incomplete
                if len(sent) < 3:
                    continue

                # Extract POS sequence
                pos_seq = [token.pos_ for token in sent]
                pos_patterns.append(" ".join(pos_seq))

                # Extract dependency relations
                dep_seq = [token.dep_ for token in sent]
                dependency_patterns.append(" ".join(dep_seq))

                # Rough estimate of clauses (based on verbs)
                # More sophisticated clause detection would require a constituency parser
                verbs = [token for token in sent if token.pos_ in ("VERB", "AUX")]
                clause_counts.append(len(verbs))

        # Calculate voice usage (active vs. passive)
        passive_count = 0
        active_count = 0

        for doc in nlp_docs:
            for sent in doc.sents:
                # A simple heuristic for passive voice detection
                if any(token.dep_ == "auxpass" for token in sent):
                    passive_count += 1
                else:
                    active_count += 1

        total_sentences = passive_count + active_count
        passive_ratio = passive_count / total_sentences if total_sentences > 0 else 0

        # Calculate average clauses per sentence
        avg_clauses = np.mean(clause_counts) if clause_counts else 0
    except (AttributeError, TypeError) as e:
        # Fallback if we're using the simple NLP implementation
        print(f"Using simplified syntactic analysis due to: {e}")
        # Provide default values for syntactic analysis
        avg_clauses = 0
        passive_ratio = 0
        pos_patterns = []
        dependency_patterns = []

    return {
        "avg_sentence_length": np.mean(sentence_lengths) if sentence_lengths else 0,
        "median_sentence_length": (
            np.median(sentence_lengths) if sentence_lengths else 0
        ),
        "sentence_length_std": np.std(sentence_lengths) if sentence_lengths else 0,
        "sentence_length_dist": sentence_lengths,
        "avg_clauses_per_sentence": avg_clauses,
        "common_pos_patterns": Counter(pos_patterns).most_common(20) if pos_patterns else [],
        "common_dependency_patterns": Counter(dependency_patterns).most_common(20) if dependency_patterns else [],
        "passive_voice_ratio": passive_ratio,
    }


def calculate_readability_metrics(corpus_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate various readability metrics for the corpus.

    Args:
        corpus_data: Dictionary containing the processed corpus

    Returns:
        Dictionary of readability metrics
    """
    print("Calculating readability metrics...")

    # Combine all documents
    full_text = " ".join(corpus_data["documents"])

    # Calculate various readability metrics
    metrics = {
        "flesch_reading_ease": textstat.flesch_reading_ease(full_text),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(full_text),
        "gunning_fog": textstat.gunning_fog(full_text),
        "smog_index": textstat.smog_index(full_text),
        "coleman_liau_index": textstat.coleman_liau_index(full_text),
        "automated_readability_index": textstat.automated_readability_index(full_text),
        "dale_chall_readability_score": textstat.dale_chall_readability_score(
            full_text
        ),
    }

    # Interpret the Flesch Reading Ease score
    f_score = metrics["flesch_reading_ease"]
    if f_score >= 90:
        f_interpretation = "Very Easy - 5th grade level"
    elif f_score >= 80:
        f_interpretation = "Easy - 6th grade level"
    elif f_score >= 70:
        f_interpretation = "Fairly Easy - 7th grade level"
    elif f_score >= 60:
        f_interpretation = "Standard - 8th-9th grade level"
    elif f_score >= 50:
        f_interpretation = "Fairly Difficult - 10th-12th grade level"
    elif f_score >= 30:
        f_interpretation = "Difficult - College level"
    else:
        f_interpretation = "Very Difficult - College graduate level"

    metrics["flesch_interpretation"] = f_interpretation

    return metrics


def analyze_discourse_markers(corpus_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze discourse markers and communication style elements.

    Args:
        corpus_data: Dictionary containing the processed corpus

    Returns:
        Dictionary of discourse features including transition markers,
        hedges, boosters, and engagement strategies
    """
    print("Analyzing discourse and pragmatic markers...")

    full_text = " ".join(corpus_data["documents"]).lower()
    sentences = corpus_data["sentences"]

    # Lists of various discourse markers
    transition_markers = {
        "contrast": [
            "however",
            "nevertheless",
            "but",
            "yet",
            "on the other hand",
            "in contrast",
            "conversely",
        ],
        "addition": [
            "furthermore",
            "moreover",
            "in addition",
            "additionally",
            "also",
            "besides",
            "what is more",
        ],
        "cause": [
            "therefore",
            "thus",
            "consequently",
            "hence",
            "as a result",
            "because of this",
            "for this reason",
        ],
        "sequence": [
            "first",
            "second",
            "third",
            "finally",
            "next",
            "then",
            "subsequently",
            "afterward",
        ],
        "example": [
            "for example",
            "for instance",
            "to illustrate",
            "such as",
            "namely",
            "specifically",
        ],
    }

    hedges = [
        "may",
        "might",
        "could",
        "perhaps",
        "possibly",
        "seems",
        "appears to be",
        "suggests",
        "indicates",
        "somewhat",
        "relatively",
        "generally",
        "usually",
        "often",
        "sometimes",
        "frequently",
        "occasionally",
        "rarely",
        "seldom",
    ]

    boosters = [
        "clearly",
        "obviously",
        "certainly",
        "definitely",
        "strongly",
        "undoubtedly",
        "without doubt",
        "demonstrates",
        "proves",
        "shows",
        "always",
        "never",
        "absolutely",
        "completely",
        "entirely",
        "fully",
    ]

    engagement_markers = [
        "note that",
        "consider",
        "imagine",
        "let us",
        "we can see",
        "it is important",
        "as you can see",
        "remember that",
        "observe",
        "look at",
        "think about",
        "as we have seen",
        "take the example of",
    ]

    self_references = ["i ", "my ", "mine ", "myself "]
    reader_references = ["you ", "your ", "yours ", "yourself "]
    inclusive_references = ["we ", "our ", "us ", "ourselves "]

    # Count occurrences of each marker type
    transition_counts = {}
    for category, markers in transition_markers.items():
        category_count = sum(full_text.count(marker) for marker in markers)
        transition_counts[category] = category_count

    hedge_counts = {hedge: full_text.count(f" {hedge} ") for hedge in hedges}
    booster_counts = {booster: full_text.count(f" {booster} ") for booster in boosters}
    engagement_counts = {
        marker: full_text.count(marker) for marker in engagement_markers
    }

    # Calculate pronoun usage
    self_ref_count = sum(full_text.count(ref) for ref in self_references)
    reader_ref_count = sum(full_text.count(ref) for ref in reader_references)
    inclusive_ref_count = sum(full_text.count(ref) for ref in inclusive_references)

    # Count questions and directives
    question_count = sum(1 for sent in sentences if sent.strip().endswith("?"))
    exclamation_count = sum(1 for sent in sentences if sent.strip().endswith("!"))

    # Normalize by total sentence count
    total_sentences = len(sentences)
    question_ratio = question_count / total_sentences if total_sentences > 0 else 0
    exclamation_ratio = (
        exclamation_count / total_sentences if total_sentences > 0 else 0
    )

    # Calculate hedge-to-booster ratio
    total_hedges = sum(hedge_counts.values())
    total_boosters = sum(booster_counts.values())
    hedge_booster_ratio = (
        total_hedges / total_boosters if total_boosters > 0 else float("inf")
    )

    return {
        "transition_marker_usage": transition_counts,
        "hedge_usage": hedge_counts,
        "booster_usage": booster_counts,
        "engagement_marker_usage": engagement_counts,
        "question_ratio": question_ratio,
        "exclamation_ratio": exclamation_ratio,
        "self_reference_count": self_ref_count,
        "reader_reference_count": reader_ref_count,
        "inclusive_reference_count": inclusive_ref_count,
        "total_hedges": total_hedges,
        "total_boosters": total_boosters,
        "hedge_booster_ratio": hedge_booster_ratio,
    }


def perform_topic_modeling(corpus_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Identify key topics in the expert's writing using LDA topic modeling.

    Args:
        corpus_data: Dictionary containing the processed corpus

    Returns:
        Dictionary containing LDA model and extracted topics
    """
    print("Performing topic modeling...")

    documents = corpus_data["documents"]

    # Tokenize and prepare for topic modeling
    tokenized_docs = []
    for doc in documents:
        tokens = word_tokenize(doc.lower())
        # Remove stopwords and non-alphabetic tokens
        stop_words = set(stopwords.words("english"))
        filtered_tokens = [
            w for w in tokens if w.isalpha() and w not in stop_words and len(w) > 2
        ]
        tokenized_docs.append(filtered_tokens)

    # Create dictionary and corpus for LDA
    dictionary = Dictionary(tokenized_docs)

    # Filter out extremely rare and common terms
    dictionary.filter_extremes(no_below=2, no_above=0.8)

    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

    # Determine appropriate number of topics based on corpus size
    num_topics = min(10, max(3, len(documents) // 2))

    # Train LDA model
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=10,
        alpha="auto",
        random_state=42,
    )

    # Extract topics
    topics = []
    for topic_id in range(num_topics):
        topic_words = lda_model.show_topic(topic_id, topn=10)
        topics.append(
            {
                "topic_id": topic_id,
                "words": [(word, round(prob, 4)) for word, prob in topic_words],
            }
        )

    return {"num_topics": num_topics, "topics": topics}


def perform_liwc_analysis(corpus_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Analyze psychological dimensions using LIWC-like word categories.

    Args:
        corpus_data: Dictionary containing the processed corpus

    Returns:
        Dictionary of psychological dimension scores
    """
    print("Performing LIWC-like psychological analysis...")

    # Define word categories (simplified version of LIWC)
    psychological_categories = {
        "positive_emotion": [
            "good",
            "great",
            "excellent",
            "wonderful",
            "amazing",
            "love",
            "happy",
            "joy",
            "beautiful",
            "pleasant",
            "enjoy",
            "nice",
            "positive",
            "fantastic",
            "glad",
            "appreciate",
            "delighted",
            "excited",
            "thrilled",
            "pleased",
        ],
        "negative_emotion": [
            "bad",
            "terrible",
            "awful",
            "poor",
            "disappointing",
            "sad",
            "unhappy",
            "hate",
            "dislike",
            "angry",
            "upset",
            "unfortunate",
            "negative",
            "worse",
            "sorry",
            "problem",
            "difficult",
            "trouble",
            "fault",
            "wrong",
        ],
        "anxiety": [
            "worry",
            "nervous",
            "anxious",
            "fear",
            "afraid",
            "scared",
            "concern",
            "stress",
            "distress",
            "tension",
            "uneasy",
            "apprehensive",
            "dread",
            "panic",
            "frightened",
            "alarm",
            "disturbed",
            "threat",
            "terrified",
        ],
        "certainty": [
            "always",
            "never",
            "absolutely",
            "certainly",
            "definitely",
            "clearly",
            "obvious",
            "undoubtedly",
            "unquestionably",
            "indisputable",
            "sure",
            "guaranteed",
            "undeniable",
            "precise",
            "exact",
            "evident",
            "proven",
        ],
        "analytical_thinking": [
            "analyze",
            "analysis",
            "evaluate",
            "assessment",
            "examine",
            "investigation",
            "research",
            "study",
            "evidence",
            "data",
            "conclusion",
            "therefore",
            "thus",
            "hypothesis",
            "theory",
            "concept",
            "principle",
            "methodology",
            "compare",
        ],
        "social_processes": [
            "talk",
            "share",
            "discuss",
            "communicate",
            "conversation",
            "message",
            "social",
            "friend",
            "family",
            "people",
            "person",
            "group",
            "team",
            "community",
            "society",
            "culture",
            "relationship",
            "interaction",
        ],
    }

    # Process the text
    full_text = " ".join(corpus_data["documents"]).lower()
    words = word_tokenize(full_text)
    word_count = len(words)

    # Count occurrences of each category
    category_scores = {}
    for category, word_list in psychological_categories.items():
        # Count occurrences and normalize by total word count
        count = sum(words.count(word) for word in word_list)
        category_scores[category] = (count / word_count) * 100 if word_count > 0 else 0

    # Calculate aggregate scores
    category_scores["emotion_ratio"] = (
        category_scores["positive_emotion"] / category_scores["negative_emotion"]
        if category_scores["negative_emotion"] > 0
        else float("inf")
    )

    return category_scores


def extract_representative_examples(
    corpus_data: Dict[str, Any], profile: Dict[str, Any], num_examples: int = 5
) -> List[str]:
    """
    Extract sentences that best represent the expert's typical style.

    Args:
        corpus_data: Dictionary containing the processed corpus
        profile: Stylometric profile generated from analysis
        num_examples: Number of examples to extract

    Returns:
        List of representative sentences
    """
    print("Extracting representative examples...")

    sentences = corpus_data["sentences"]

    # Filter for sentences of appropriate length (not too short, not too long)
    avg_len = profile["syntactic_patterns"]["avg_sentence_length"]
    min_len = max(10, int(avg_len * 0.7))  # Set minimum length
    max_len = min(50, int(avg_len * 1.8))  # Set maximum length

    filtered_sentences = [
        s
        for s in sentences
        if len(s.split()) >= min_len
        and len(s.split()) <= max_len
        and not s.strip().endswith("?")  # Avoid questions
        and not s.strip().endswith("!")
    ]  # Avoid exclamations

    # If we have too few sentences, relax the constraints
    if len(filtered_sentences) < num_examples * 2:
        filtered_sentences = [s for s in sentences if len(s.split()) >= 5]

    # Score sentences based on how well they match the profile
    scored_sentences = []

    for sentence in filtered_sentences:
        score = 0
        sentence_lower = sentence.lower()

        # Check for common content words
        common_words = [
            word for word, _ in profile["lexical_features"]["common_words"][:30]
        ]
        word_matches = sum(1 for word in common_words if word in sentence_lower.split())
        score += word_matches * 0.5

        # Check for characteristic n-grams
        for bigram, _ in profile["lexical_features"]["common_bigrams"][:10]:
            if " ".join(bigram) in sentence_lower:
                score += 1

        # Check for discourse markers that are characteristic
        for marker, count in profile["discourse_analysis"]["hedge_usage"].items():
            if marker in sentence_lower:
                score += 0.5

        for marker, count in profile["discourse_analysis"]["booster_usage"].items():
            if marker in sentence_lower:
                score += 0.5

        # Check for transition markers
        for category, count in profile["discourse_analysis"][
            "transition_marker_usage"
        ].items():
            if count > 5:  # Only consider frequent markers
                for marker in [
                    "however",
                    "therefore",
                    "furthermore",
                    "moreover",
                    "thus",
                ]:
                    if marker in sentence_lower:
                        score += 0.5

        # Penalize quoted content slightly as it might not be the expert's own words
        if '"' in sentence or "'" in sentence:
            score -= 0.5

        # Add the sentence with its score
        scored_sentences.append((sentence, score))

    # Return the highest-scoring sentences, ensuring they're different from each other
    sorted_sentences = sorted(scored_sentences, key=lambda x: x[1], reverse=True)

    # Choose diverse examples by ensuring they don't share too many words
    selected = []
    for sentence, score in sorted_sentences:
        # Skip very similar sentences
        if any(get_text_similarity(sentence, s) > 0.5 for s in selected):
            continue

        selected.append(sentence)
        if len(selected) >= num_examples:
            break

    return selected


def extract_prompt_template(profile: Dict[str, Any]) -> str:
    """
    Generate a prompt template for LLM persona styling based on the profile.

    Args:
        profile: Dictionary containing the stylometric profile

    Returns:
        A formatted prompt template string
    """
    print("Generating LLM prompt template...")

    # Extract key style elements
    avg_sentence_length = profile["syntactic_patterns"]["avg_sentence_length"]
    flesch_score = profile["readability_metrics"]["flesch_reading_ease"]
    hedge_booster_ratio = profile["discourse_analysis"]["hedge_booster_ratio"]
    passive_ratio = profile["syntactic_patterns"]["passive_voice_ratio"]

    # Get word preferences
    common_words = [
        word for word, _ in profile["lexical_features"]["common_words"][:15]
    ]
    common_bigrams = [
        " ".join(bigram)
        for bigram, _ in profile["lexical_features"]["common_bigrams"][:8]
    ]

    # Get common hedges or boosters based on which is used more
    hedges = list(profile["discourse_analysis"]["hedge_usage"].keys())
    boosters = list(profile["discourse_analysis"]["booster_usage"].keys())

    # Determine persona characteristics
    if flesch_score < 40:
        formality = "formal, academic"
        audience = "educated specialists"
    elif flesch_score < 60:
        formality = "semi-formal, professional"
        audience = "informed professionals or enthusiasts"
    else:
        formality = "conversational, accessible"
        audience = "general readers"

    if hedge_booster_ratio > 1.5:
        assertion_style = "tentative, thoughtful"
        modifiers = f"frequently use hedging phrases like {', '.join(hedges[:3])}"
    elif hedge_booster_ratio < 0.7:
        assertion_style = "confident, authoritative"
        modifiers = f"use boosting phrases like {', '.join(boosters[:3])}"
    else:
        assertion_style = "balanced"
        modifiers = "balance tentative and assertive language"

    # Pronoun usage
    self_refs = profile["discourse_analysis"]["self_reference_count"]
    reader_refs = profile["discourse_analysis"]["reader_reference_count"]
    inclusive_refs = profile["discourse_analysis"]["inclusive_reference_count"]

    if self_refs > reader_refs and self_refs > inclusive_refs:
        pronouns = "frequent first-person perspective (I, my, mine)"
    elif inclusive_refs > self_refs and inclusive_refs > reader_refs:
        pronouns = "inclusive we/our perspective to build connection"
    elif reader_refs > self_refs:
        pronouns = "direct address to the reader (you, your)"
    else:
        pronouns = "balanced mix of perspectives"

    # Sentence structure guidance
    if avg_sentence_length < 15:
        sentence_structure = (
            f"Short, direct sentences (avg {avg_sentence_length:.1f} words)"
        )
    elif avg_sentence_length < 25:
        sentence_structure = (
            f"Medium-length sentences (avg {avg_sentence_length:.1f} words)"
        )
    else:
        sentence_structure = (
            f"Longer, more complex sentences (avg {avg_sentence_length:.1f} words)"
        )

    # Transition style
    transition_types = profile["discourse_analysis"]["transition_marker_usage"]
    top_transition = max(transition_types.items(), key=lambda x: x[1])

    # Build template
    template = f"""# PERSONA DEFINITION
You are embodying the persona of {{EXPERT_NAME}}, a {{FIELD}} expert. 

# STYLISTIC PARAMETERS
## Voice and Tone
- Formality: {formality}
- Target Audience: {audience}
- Assertion Style: {assertion_style}
- Language Modifiers: {modifiers}
- Perspective: {pronouns}

## Structural Elements
- Sentence Structure: {sentence_structure}"""

    if passive_ratio > 0.2:
        template += f"\n- Voice: Use passive voice in about {passive_ratio*100:.1f}% of sentences"
    else:
        template += "\n- Voice: Prefer active voice"

    template += f"\n- Transitions: Favor '{top_transition[0]}' transitions"

    template += """
- Paragraph Organization: Maintain moderate paragraph length with clear topic sentences

## Vocabulary and Phrasing
- Preferred Terminology:"""

    for word in common_words[:10]:
        template += f"\n  * {word}"

    template += "\n- Characteristic Phrases:"
    for phrase in common_bigrams[:5]:
        template += f"\n  * {phrase}"

    template += """

# AUTHENTIC EXAMPLES
The following examples demonstrate the expert's authentic writing style:
"""

    for example in profile["representative_examples"]:
        template += f'\nEXAMPLE: "{example}"\n'

    template += """
# TASK
A user has asked the following question. Respond as {{EXPERT_NAME}} would, maintaining their authentic voice, knowledge perspective, and communication style as defined above.

USER QUESTION: {{USER_QUESTION}}
"""

    return template


def get_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two texts using Jaccard similarity of words.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Similarity score between 0 and 1
    """
    words1 = set(word_tokenize(text1.lower()))
    words2 = set(word_tokenize(text2.lower()))

    # Jaccard similarity: intersection over union
    intersection = words1.intersection(words2)
    union = words1.union(words2)

    return len(intersection) / len(union) if union else 0


def create_summary_report(profile: Dict[str, Any], output_file: str):
    """
    Create a human-readable summary of the stylometric profile.

    Args:
        profile: Dictionary containing the stylometric profile
        output_file: File path to save the summary report
    """
    print(f"Creating summary report at {output_file}...")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("EXPERT STYLOMETRIC PROFILE\n")
        f.write("=========================\n\n")

        # Lexical summary
        f.write("## LEXICAL STYLE\n")
        f.write(f"Vocabulary size: {profile['lexical_features']['vocabulary_size']}\n")
        f.write(
            f"Type-token ratio: {profile['lexical_features']['type_token_ratio']:.4f}\n"
        )
        f.write(
            f"Average word length: {profile['lexical_features']['average_word_length']:.2f} characters\n"
        )

        f.write("\nMost common content words:\n")
        for word, count in profile["lexical_features"]["common_words"][:20]:
            f.write(f"  - {word}: {count}\n")

        f.write("\nMost distinctive function words (compared to general usage):\n")
        for word, freq in sorted(
            profile["lexical_features"]["function_word_dist"].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]:
            f.write(f"  - {word}: {freq*100:.2f}%\n")

        # Syntactic summary
        f.write("\n## SYNTACTIC PATTERNS\n")
        f.write(
            f"Average sentence length: {profile['syntactic_patterns']['avg_sentence_length']:.2f} words\n"
        )
        f.write(
            f"Median sentence length: {profile['syntactic_patterns']['median_sentence_length']:.2f} words\n"
        )
        f.write(
            f"Sentence length standard deviation: {profile['syntactic_patterns']['sentence_length_std']:.2f}\n"
        )
        f.write(
            f"Average clauses per sentence: {profile['syntactic_patterns']['avg_clauses_per_sentence']:.2f}\n"
        )
        f.write(
            f"Passive voice usage: {profile['syntactic_patterns']['passive_voice_ratio']*100:.1f}%\n"
        )

        # Readability
        f.write("\n## READABILITY\n")
        f.write(
            f"Flesch Reading Ease: {profile['readability_metrics']['flesch_reading_ease']:.1f} "
        )
        f.write(f"({profile['readability_metrics']['flesch_interpretation']})\n")
        f.write(
            f"Flesch-Kincaid Grade Level: {profile['readability_metrics']['flesch_kincaid_grade']:.1f}\n"
        )
        f.write(
            f"Gunning Fog Index: {profile['readability_metrics']['gunning_fog']:.1f}\n"
        )
        f.write(f"SMOG Index: {profile['readability_metrics']['smog_index']:.1f}\n")
        f.write(
            f"Coleman-Liau Index: {profile['readability_metrics']['coleman_liau_index']:.1f}\n"
        )

        # Discourse style
        f.write("\n## DISCOURSE STYLE\n")
        f.write(
            f"Question frequency: {profile['discourse_analysis']['question_ratio']*100:.1f}% of sentences\n"
        )
        f.write(
            f"Exclamation frequency: {profile['discourse_analysis']['exclamation_ratio']*100:.1f}% of sentences\n"
        )

        f.write("\nReferencing style:\n")
        f.write(
            f"  - Self references (I, my, etc.): {profile['discourse_analysis']['self_reference_count']}\n"
        )
        f.write(
            f"  - Reader references (you, your, etc.): {profile['discourse_analysis']['reader_reference_count']}\n"
        )
        f.write(
            f"  - Inclusive references (we, our, etc.): {profile['discourse_analysis']['inclusive_reference_count']}\n"
        )

        f.write(
            f"\nHedging vs. Boosting: {profile['discourse_analysis']['hedge_booster_ratio']:.2f} ratio\n"
        )
        f.write(
            f"  (Higher values indicate more tentative language; lower values indicate more assertive language)\n"
        )

        f.write("\nTransition marker usage by category:\n")
        for category, count in profile["discourse_analysis"][
            "transition_marker_usage"
        ].items():
            f.write(f"  - {category}: {count}\n")

        # Psychological dimensions
        f.write("\n## PSYCHOLOGICAL DIMENSIONS\n")
        for dimension, score in profile["liwc_analysis"].items():
            if dimension != "emotion_ratio":
                f.write(f"{dimension.replace('_', ' ').title()}: {score:.2f}%\n")

        f.write(
            f"Positive-to-Negative emotion ratio: {profile['liwc_analysis']['emotion_ratio']:.2f}\n"
        )

        # Topics
        f.write("\n## COMMON TOPICS\n")
        for topic in profile["topic_model"]["topics"]:
            topic_terms = ", ".join(
                f"{word} ({prob:.3f})" for word, prob in topic["words"]
            )
            f.write(f"Topic {topic['topic_id']}: {topic_terms}\n")

        # Representative examples
        f.write("\n## REPRESENTATIVE EXAMPLES\n")
        for i, example in enumerate(profile["representative_examples"], 1):
            f.write(f'Example {i}: "{example}"\n\n')

        # Style guidelines for prompting
        f.write("\n## STYLE GUIDELINES FOR LLM PROMPTING\n")
        f.write(
            "Based on the analysis, here are key style elements to include in prompts:\n\n"
        )

        # Sentence structure guidance
        avg_len = profile["syntactic_patterns"]["avg_sentence_length"]
        if avg_len < 15:
            f.write("1. Use predominantly short sentences (around ")
            f.write(f"{avg_len:.0f} words on average)\n")
        elif avg_len < 25:
            f.write("1. Use a mix of medium-length sentences (around ")
            f.write(f"{avg_len:.0f} words on average)\n")
        else:
            f.write("1. Use longer, more complex sentences (around ")
            f.write(f"{avg_len:.0f} words on average)\n")

        # Formality guidance based on readability
        f_score = profile["readability_metrics"]["flesch_reading_ease"]
        if f_score < 40:
            f.write("2. Use a formal, academic tone with sophisticated vocabulary\n")
        elif f_score < 60:
            f.write("2. Use a semi-formal tone with professional vocabulary\n")
        else:
            f.write("2. Use a conversational, accessible tone\n")

        # Assertiveness based on hedge-booster ratio
        h_b_ratio = profile["discourse_analysis"]["hedge_booster_ratio"]
        if h_b_ratio > 2:
            f.write("3. Express ideas tentatively, using hedges like ")
            hedges = list(profile["discourse_analysis"]["hedge_usage"].keys())
            f.write(f"'{hedges[0]}', '{hedges[1]}', and '{hedges[2]}'\n")
        elif h_b_ratio < 0.5:
            f.write("3. Express ideas confidently, using boosters like ")
            boosters = list(profile["discourse_analysis"]["booster_usage"].keys())
            f.write(f"'{boosters[0]}', '{boosters[1]}', and '{boosters[2]}'\n")
        else:
            f.write("3. Balance tentative and assertive language\n")

        # Personal pronouns
        if (
            profile["discourse_analysis"]["self_reference_count"]
            > profile["discourse_analysis"]["reader_reference_count"]
        ):
            f.write("4. Use first-person perspective frequently (I, my, etc.)\n")
        elif (
            profile["discourse_analysis"]["inclusive_reference_count"]
            > profile["discourse_analysis"]["self_reference_count"]
        ):
            f.write("4. Use inclusive first-person plural frequently (we, our, etc.)\n")
        elif (
            profile["discourse_analysis"]["reader_reference_count"]
            > profile["discourse_analysis"]["self_reference_count"]
        ):
            f.write("4. Address the reader directly (you, your, etc.)\n")

        # Characteristic words
        f.write("5. Incorporate these characteristic words and phrases:\n")
        common_words = [
            word for word, _ in profile["lexical_features"]["common_words"][:10]
        ]
        f.write(f"   - Content words: {', '.join(common_words)}\n")

        # Add bigrams as characteristic phrases
        common_bigrams = [
            " ".join(bigram)
            for bigram, _ in profile["lexical_features"]["common_bigrams"][:5]
        ]
        f.write(f"   - Phrases: {', '.join(common_bigrams)}\n")

        # Transition style
        top_transition = max(
            profile["discourse_analysis"]["transition_marker_usage"].items(),
            key=lambda x: x[1],
        )
        f.write(f"6. Use '{top_transition[0]}' transitions frequently\n")


def generate_persona_profile(
    corpus_files: List[str], output_prefix: str = "expert_profile"
) -> Dict[str, Any]:
    """
    Generate a complete stylometric profile from a corpus of texts.

    Args:
        corpus_files: List of files in the corpus
        output_prefix: Prefix for output files

    Returns:
        Complete stylometric profile dictionary
    """
    print(f"Generating persona profile from {len(corpus_files)} files...")

    # Preprocess the corpus
    corpus_data = preprocess_corpus(corpus_files)

    # Extract all features
    lexical_features = extract_lexical_features(corpus_data)
    syntactic_patterns = analyze_syntactic_patterns(corpus_data)
    readability_metrics = calculate_readability_metrics(corpus_data)
    discourse_analysis = analyze_discourse_markers(corpus_data)
    topic_model = perform_topic_modeling(corpus_data)
    liwc_analysis = perform_liwc_analysis(corpus_data)

    # Compile the initial profile
    profile = {
        "lexical_features": lexical_features,
        "syntactic_patterns": syntactic_patterns,
        "readability_metrics": readability_metrics,
        "discourse_analysis": discourse_analysis,
        "topic_model": topic_model,
        "liwc_analysis": liwc_analysis,
    }

    # Extract representative examples
    representative_examples = extract_representative_examples(
        corpus_data, profile, num_examples=5
    )
    profile["representative_examples"] = representative_examples

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_prefix)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Export profile to JSON
    json_file = f"{output_prefix}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        # Convert numpy values to Python native types for JSON serialization
        def serialize_json(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Type {type(obj)} not serializable")

        json.dump(profile, f, indent=2, default=serialize_json, ensure_ascii=False)
    print(f"Profile data saved to {json_file}")

    # Create summary report
    report_file = f"{output_prefix}_summary.txt"
    create_summary_report(profile, report_file)
    print(f"Summary report saved to {report_file}")

    # Create visualizations
    viz_dir = f"{output_prefix}_visualizations"
    create_visualizations(profile, viz_dir)
    print(f"Visualizations saved to {viz_dir}")

    # Generate LLM prompt template
    prompt_template = extract_prompt_template(profile)
    prompt_file = f"{output_prefix}_prompt_template.txt"
    with open(prompt_file, "w", encoding="utf-8") as f:
        f.write(prompt_template)
    print(f"LLM prompt template saved to {prompt_file}")

    return profile


def create_visualizations(profile: Dict[str, Any], output_dir: str):
    """
    Create visualizations of the stylometric profile.

    Args:
        profile: Dictionary containing the stylometric profile
        output_dir: Directory to save visualizations
    """
    print("Creating visualizations...")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Set seaborn style
    sns.set(style="whitegrid")

    # 1. Sentence length distribution
    plt.title("Psychological Dimensions (% of words)")
    plt.xlabel("Percentage")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "psychological_dimensions.png"))
    plt.close()

    # 6. Readability metrics
    plt.figure(figsize=(10, 6))
    readability_metrics = [
        "flesch_reading_ease",
        "flesch_kincaid_grade",
        "gunning_fog",
        "smog_index",
        "coleman_liau_index",
        "automated_readability_index",
    ]
    readability_values = [
        profile["readability_metrics"][m] for m in readability_metrics
    ]

    sns.barplot(x=readability_values, y=readability_metrics)
    plt.title("Readability Metrics")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "readability_metrics.png"))
    plt.close()


def main():
    """Main function to run the stylometric profiler."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate a stylometric profile from a corpus of text."
    )
    parser.add_argument(
        "--corpus_dir", type=str, help="Directory containing corpus files"
    )
    parser.add_argument(
        "--corpus_files", type=str, nargs="+", help="Specific corpus files to analyze"
    )
    parser.add_argument(
        "--output", type=str, default="expert_profile", help="Output file prefix"
    )
    args = parser.parse_args()

    # Get list of files to analyze
    corpus_files = []
    if args.corpus_dir:
        # Walk through directory and get all text files
        for root, dirs, files in os.walk(args.corpus_dir):
            for file in files:
                if file.endswith((".txt", ".md", ".csv", ".html", ".xml", ".json")):
                    corpus_files.append(os.path.join(root, file))
    elif args.corpus_files:
        corpus_files = args.corpus_files
    else:
        print("Error: Either --corpus_dir or --corpus_files must be specified.")
        parser.print_help()
        return

    if not corpus_files:
        print("No corpus files found. Please check the directory or file paths.")
        return

    print(f"Found {len(corpus_files)} files to analyze")

    # Generate the profile
    profile = generate_persona_profile(corpus_files, args.output)

    print("\nProfile generation complete!")
    print(f"Results saved with prefix: {args.output}")


if __name__ == "__main__":
    main()
