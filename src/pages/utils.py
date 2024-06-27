import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize
import random
from dash import html
import numpy as np

nltk.download('stopwords', quiet=True)
nltk.download('brown', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

stop_words = set(stopwords.words('english'))

def filter_text(text):
    allowed_pos = ['NN', 'NNP', 'NNS', 'VB']
    tokens = word_tokenize(text)
    tokens_pos = nltk.pos_tag(tokens)
    filtered_text = [x[0] for x in tokens_pos if x[1] in allowed_pos and x[0] not in stop_words]
    return filtered_text

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.name() != word:
                synonyms.add(lemma.name())
    return list(synonyms)


def add_synonyms(string):
    tokens = word_tokenize(string)
    syn_out = []

    for token in tokens:
        try:
            synonyms = get_synonyms(token)
        except:
            synonyms = []

        if synonyms:
            # Pick 5 random synonyms from the list, or fewer if less than 5 are available
            random_synonyms = random.sample(synonyms, min(5, len(synonyms)))
            # Create a span element with tooltip and apply styling
            token_element = html.Span(
                token,
                className='synonym-token',  # CSS class for styling
                title=f'Synonyms: {", ".join(random_synonyms)}'  # Tooltip content
            )
            syn_out.append(token_element)
            syn_out.append(" ")  # Add space between tokens for readability
        else:
            syn_out.append(token + " ")  # No tooltip if no synonyms found

    return syn_out


def get_attention_colored_text(text, attention_scores):
    tokens = word_tokenize(text)
    attention_scores = np.array(attention_scores)
    min_score, max_score = attention_scores.min(), attention_scores.max()
    normalized_scores = (attention_scores - min_score) / (max_score - min_score)

    colored_text = ""
    for token, score in zip(tokens, normalized_scores):
        color = f"rgb({int(255 * (1 - score))}, {int(255 * score)}, 0)"  # Gradient from red to green
        colored_text += f'<span style="color: {color};">{token} </span>'

    return colored_text

def generate_random_attention_scores(text):
    tokens = word_tokenize(text)
    random_scores = np.random.rand(len(tokens))
    return random_scores


# This function returns HTML element that shows attention in color.
# @callback(
#     Output('colored-text-output', 'children'),
#     Input('run-all-prompts-btn', 'n_clicks')
# )
def display_colored_text(n_clicks):
    if n_clicks is None:
        return ""
    
    # Currently we use this sample example, replace this with a prompt
    input_text = "The quick brown fox jumps over the lazy dog."

    
    attention_scores = generate_random_attention_scores(input_text)
    tokens = word_tokenize(input_text)
    normalized_scores = (attention_scores - attention_scores.min()) / (attention_scores.max() - attention_scores.min())

    colored_text = []
    for token, score in zip(tokens, normalized_scores):
        color = f"rgb({int(255 * (1 - score))}, {int(255 * score)}, 0)"  # Gradient from red to green
        colored_text.append(
            html.Span(token + " ", style={"color": color})
        )
    
    return colored_text

# def preprocess_text(text):
#     # Tokenize the text into words
#     tokenizer = RegexpTokenizer(r'\w+')
#     tokenized_1 = tokenizer.tokenize(text.lower())
#     words = word_tokenize(' '.join(tokenized_1))

#     # Define the stopwords and punctuation
#     stop_words = set(stopwords.words('english'))

#     # Remove stopwords and punctuation
#     tokens = [word for word in words if word not in stop_words]

#     return ' '.join(tokens)


def preprocess_text(text):
    # Tokenize the text into words
    words = word_tokenize(text.lower())

    # Define the stopwords and punctuation
    stop_words = set(stopwords.words('english'))

    # Remove stopwords and punctuation
    tokens = [word for word in words if word not in stop_words]

    return ' '.join(tokens)
