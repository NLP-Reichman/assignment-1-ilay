# -*- coding: utf-8 -*-
"""Assignment 1

This assignment implements tools for learning and testing language models.
The corpora that we work with are lists of tweets in 8 different languages that use the Latin script.
The data is provided either formatted as CSV or as JSON.
The end goal is to write a set of tools that can detect the language of a given tweet.
"""

import os
import json
import pandas as pd
import numpy as np
from itertools import product

languages = ['en', 'es', 'fr', 'in', 'it', 'nl', 'pt', 'tl']

def preprocess() -> list[str]:
  '''
  Return a list of characters, representing the shared vocabulary of all languages
  '''
  vocab = set(['<start>', '<end>', '<unk>'])  # Add special tokens
  languages = ['en', 'es', 'fr', 'in', 'it', 'nl', 'pt', 'tl']
  
  for lang in languages:
    # Try CSV first, if not found try JSON
    try:
      df = pd.read_csv(f'data/{lang}.csv')
      texts = df['text'].astype(str).tolist()
    except:
      with open(f'data/{lang}.json', 'r') as f:
        data = json.load(f)
        texts = [str(item['text']) for item in data]
    
    # Process each text and add characters to vocabulary
    for text in texts:
      text = str(text)  # Ensure string type
      vocab.update(set(text))
  
  return sorted(list(vocab))

def build_lm(lang: str, n: int, smoothed: bool = False) -> dict[str, dict[str, float]]:
  '''
  Return a language model for the given lang and n_gram (n)
  :param lang: the language of the model
  :param n: the n_gram value
  :param smoothed: boolean indicating whether to apply smoothing
  :return: a dictionary where the keys are n_grams and the values are dictionaries
  '''
  # Initialize counters
  ngram_counts = {}
  context_counts = {}
  vocab = preprocess()
  V = len(vocab)  # Vocabulary size for smoothing
  
  # Read data
  try:
    df = pd.read_csv(f'data/{lang}.csv')
    texts = df['text'].astype(str).tolist()
  except:
    with open(f'data/{lang}.json', 'r') as f:
      data = json.load(f)
      texts = [str(item['text']) for item in data]
  
  # Process each text
  for text in texts:
    text = str(text)  # Ensure string type
    padded_text = '<start>' * (n-1) + text + '<end>'
    
    # Count n-grams
    for i in range(len(padded_text) - n + 1):
      context = padded_text[i:i+n-1]
      next_char = padded_text[i+n-1]
      
      # Update counts
      if context not in ngram_counts:
        ngram_counts[context] = {}
      if next_char not in ngram_counts[context]:
        ngram_counts[context][next_char] = 0
      ngram_counts[context][next_char] += 1
      
      if context not in context_counts:
        context_counts[context] = 0
      context_counts[context] += 1
  
  # Calculate probabilities
  LM = {}
  for context in ngram_counts:
    LM[context] = {}
    total = context_counts[context]
    
    if smoothed:
      # Add-one smoothing
      for char in vocab:
        count = ngram_counts[context].get(char, 0) + 1
        LM[context][char] = count / (total + V)
    else:
      # No smoothing
      for char, count in ngram_counts[context].items():
        LM[context][char] = count / total
  
  return LM

def perplexity(model: dict, text: list, n: int) -> float:
  '''
  Calculates the perplexity of the given string using the given language model.
  :param model: The language model
  :param text: The tokenized text to calculate the perplexity for
  :param n: The n-gram of the model
  :return: The perplexity
  '''
  log_prob = 0
  total_chars = 0
  vocab = preprocess()
  unk_prob = 1.0 / len(vocab)  # Probability for unknown tokens
  
  for sentence in text:
    sentence = str(sentence)  # Ensure string type
    padded_text = '<start>' * (n-1) + sentence + '<end>'
    total_chars += len(padded_text) - (n-1)  # Subtract start tokens
    
    # Calculate log probability
    for i in range(len(padded_text) - n + 1):
      context = padded_text[i:i+n-1]
      next_char = padded_text[i+n-1]
      
      # Get probability from model or use <unk> probability
      if context in model:
        prob = model[context].get(next_char, unk_prob)
      else:
        prob = unk_prob
      
      log_prob += np.log(prob)
  
  # Calculate perplexity
  pp = np.exp(-log_prob / total_chars)
  return pp

def eval(model: dict, target_lang: str, n: int) -> float:
  '''
  Return the perplexity value calculated over applying the model on the text file
  of the target_lang language.
  :param model: the language model
  :param target_lang: the target language
  :param n: The n-gram of the model
  :return: the perplexity value
  '''
  # Read data
  try:
    df = pd.read_csv(f'data/{target_lang}.csv')
    texts = df['text'].astype(str).tolist()
  except:
    with open(f'data/{target_lang}.json', 'r') as f:
      data = json.load(f)
      texts = [str(item['text']) for item in data]  # Convert to string
  
  return perplexity(model, texts, n)

def match() -> pd.DataFrame:
  '''
  Return a DataFrame containing one line per every language pair and n_gram.
  Each line will contain the perplexity calculated when applying the language model
  of the source language on the text of the target language.
  :return: a DataFrame containing the perplexity values
  '''
  results = []
  languages = ['en', 'es', 'fr', 'in', 'it', 'nl', 'pt', 'tl']
  
  # For each language pair and n-gram
  for source_lang in languages:
    for target_lang in languages:
      for n in range(1, 5):  # n from 1 to 4
        # Build model and calculate perplexity
        model = build_lm(source_lang, n, True)
        pp = eval(model, target_lang, n)
        
        # Add result
        results.append({
          'source': source_lang,
          'target': target_lang,
          'n': n,
          'perplexity': pp
        })
  
  return pd.DataFrame(results)

def generate(lang: str, n: int, prompt: str, number_of_tokens: int, r: int) -> str:
  '''
  Generate text in the given language using the given parameters.
  :param lang: the language of the model
  :param n: the n_gram value
  :param prompt: the prompt to start the generation
  :param number_of_tokens: the number of tokens to generate
  :param r: the random seed to use
  '''
  # Set random seed
  np.random.seed(r)
  
  # Build model
  model = build_lm(lang, n, False)  # No smoothing for generation
  
  # Initialize text with prompt
  text = str(prompt)
  chars = list(text)  # Convert to list of characters
  
  # Generate tokens
  for _ in range(number_of_tokens):
    # Get context (last n-1 characters)
    if len(chars) >= n-1:
      context = ''.join(chars[-(n-1):])
    else:
      context = '<start>' * (n-1-len(chars)) + ''.join(chars)
    
    # Get next character probabilities
    if context in model:
      probs = model[context]
      # Sample next character
      next_char = np.random.choice(list(probs.keys()), p=list(probs.values()))
    else:
      # If context not in model, sample from vocabulary
      vocab = preprocess()
      next_char = np.random.choice(vocab)
    
    # Add character to text
    chars.append(next_char)
    
    # Stop if we generate end token
    if next_char == '<end>':
      break
  
  return ''.join(chars)

# Run tests and create results.json
def test_preprocess():
    return {
        'vocab_length': len(preprocess()),
    }

def test_build_lm():
    return {
        'english_2_gram_length': len(build_lm('en', 2, True)),
        'english_3_gram_length': len(build_lm('en', 3, True)),
        'french_3_gram_length': len(build_lm('fr', 3, True)),
        'spanish_3_gram_length': len(build_lm('es', 3, True)),
    }

def test_eval():
    lm = build_lm('en', 3, True)
    return {
        'en_on_en': round(eval(lm, 'en', 3), 2),
        'en_on_fr': round(eval(lm, 'fr', 3), 2),
        'en_on_tl': round(eval(lm, 'tl', 3), 2),
        'en_on_nl': round(eval(lm, 'nl', 3), 2),
    }

def test_match():
    df = match()
    return {
        'df_shape': df.shape,
        'en_en_3': df[(df['source'] == 'en') & (df['target'] == 'en') & (df['n'] == 3)]['perplexity'].values[0],
        'en_tl_3': df[(df['source'] == 'en') & (df['target'] == 'tl') & (df['n'] == 3)]['perplexity'].values[0],
        'en_nl_3': df[(df['source'] == 'en') & (df['target'] == 'nl') & (df['n'] == 3)]['perplexity'].values[0],
    }

def test_generate():
    return {
        'english_2_gram': generate('en', 2, "I am", 20, 5),
        'english_3_gram': generate('en', 3, "I am", 20, 5),
        'english_4_gram': generate('en', 4, "I Love", 20, 5),
        'spanish_2_gram': generate('es', 2, "Soy", 20, 5),
        'spanish_3_gram': generate('es', 3, "Soy", 20, 5),
        'french_2_gram': generate('fr', 2, "Je suis", 20, 5),
        'french_3_gram': generate('fr', 3, "Je suis", 20, 5),
    }

TESTS = [test_preprocess, test_build_lm, test_eval, test_match, test_generate]

# Run tests and save results
res = {}
for test in TESTS:
    try:
        cur_res = test()
        res.update({test.__name__: cur_res})
    except Exception as e:
        res.update({test.__name__: repr(e)})

with open('results.json', 'w') as f:
    json.dump(res, f, indent=2)
