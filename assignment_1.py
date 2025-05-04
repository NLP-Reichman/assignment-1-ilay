import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import random

def read_data(lang):
    """Helper function to read data for a specific language"""
    df = pd.read_csv(f'data/{lang}.csv')
    return df['tweet_text'].astype(str).tolist()

def preprocess():
    """
    Preprocess the data and create vocabulary.
    Returns the length of vocabulary.
    """
    # Read all languages
    languages = ['en', 'es', 'fr', 'in', 'it', 'nl', 'pt', 'tl']
    all_text = []
    
    for lang in languages:
        texts = read_data(lang)
        all_text.extend(texts)
    
    # Create vocabulary (unique characters)
    vocab = set()
    for text in all_text:
        vocab.update(set(text))
    
    # Add special tokens
    vocab.add('<start>')
    vocab.add('<end>')
    vocab.add('<unk>')
    
    return sorted(list(vocab))

def get_ngrams(text, n):
    """Helper function to get n-grams from text"""
    # Add start and end tokens
    text = '<start>' * (n-1) + text + '<end>'
    if len(text) < n:
        return []
    return [text[i:i+n] for i in range(len(text)-n+1)]

def build_lm(language, n, use_smoothing=False):
    """
    Build an n-gram language model for the specified language.
    
    Args:
        language (str): Language code ('en', 'fr', 'es', etc.)
        n (int): n-gram size
        use_smoothing (bool): Whether to use smoothing
    
    Returns:
        dict: Language model probabilities
    """
    texts = read_data(language)
    
    # Count n-grams and (n-1)-grams
    ngram_counts = defaultdict(int)
    context_counts = defaultdict(int)
    
    for text in texts:
        ngrams = get_ngrams(text, n)
        for ngram in ngrams:
            ngram_counts[ngram] += 1
            context = ngram[:-1]
            context_counts[context] += 1
    
    # Calculate probabilities
    lm = {}
    vocab_size = len(preprocess())
    
    for ngram, count in ngram_counts.items():
        context = ngram[:-1]
        next_char = ngram[-1]
        
        if context not in lm:
            lm[context] = {}
        
        if use_smoothing:
            # Add-1 smoothing
            prob = (count + 1) / (context_counts[context] + vocab_size)
        else:
            prob = count / context_counts[context]
        
        lm[context][next_char] = prob
    
    # Add <unk> token for smoothed models
    if use_smoothing:
        for context in lm:
            lm[context]['<unk>'] = 1.0 / vocab_size
    
    return lm

def calculate_perplexity(text, lm, n):
    """Helper function to calculate perplexity"""
    ngrams = get_ngrams(text, n)
    log_prob = 0
    count = 0
    
    for ngram in ngrams:
        context = ngram[:-1]
        next_char = ngram[-1]
        
        if context in lm:
            if next_char in lm[context]:
                prob = lm[context][next_char]
            else:
                prob = lm[context].get('<unk>', 1.0 / len(preprocess()))
            log_prob += np.log2(prob)
            count += 1
    
    if count == 0:
        return float('inf')
    
    return 2 ** (-log_prob / count)

def eval(language_model, test_language, n):
    """
    Evaluate a language model on test data.
    
    Args:
        language_model (dict): The trained language model
        test_language (str): Language code of test data
        n (int): n-gram size
    
    Returns:
        float: Perplexity score
    """
    test_texts = read_data(test_language)
    perplexities = []
    
    for text in test_texts:
        if len(text) >= n:
            pp = calculate_perplexity(text, language_model, n)
            if pp != float('inf'):
                perplexities.append(pp)
    
    return np.mean(perplexities) if perplexities else float('inf')

def match():
    """
    Match languages using the language models.
    
    Returns:
        pd.DataFrame: DataFrame with results
    """
    languages = ['en', 'es', 'fr', 'in', 'it', 'nl', 'pt', 'tl']
    results = []
    
    for source_lang in languages:
        for n in [1, 2, 3, 4]:  # Test with n-grams from 1 to 4
            lm = build_lm(source_lang, n, True)
            
            for target_lang in languages:
                perplexity = eval(lm, target_lang, n)
                results.append({
                    'source': source_lang,
                    'target': target_lang,
                    'n': n,
                    'perplexity': perplexity
                })
    
    return pd.DataFrame(results)

def generate(language, n, prefix, length, num_sentences):
    """
    Generate text using the language model.
    
    Args:
        language (str): Language code
        n (int): n-gram size
        prefix (str): Starting text
        length (int): Length of generated text
        num_sentences (int): Number of sentences to generate
    
    Returns:
        list: Generated sentences
    """
    lm = build_lm(language, n, False)  # No smoothing for generation
    generated_texts = []
    random.seed(42)  # For reproducibility
    
    for _ in range(num_sentences):
        current_text = prefix
        
        while len(current_text) < length:
            context = current_text[-(n-1):] if len(current_text) >= (n-1) else current_text
            
            if context in lm:
                next_chars = list(lm[context].keys())
                probs = list(lm[context].values())
                next_char = np.random.choice(next_chars, p=probs)
                current_text += next_char
            else:
                # If context not found, add a random character from vocabulary
                vocab = preprocess()
                current_text += random.choice(vocab)
        
        generated_texts.append(current_text)
    
    return generated_texts 