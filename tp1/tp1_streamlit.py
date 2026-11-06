"""
TP N°1 - Text Preprocessing (Streamlit version)

Install:  pip install streamlit nltk
Run with: streamlit run tp1_streamlit.py

Expected dataset layout (BBC News - Full Text):
    bbc/
        business/001.txt, 002.txt, ...
        entertainment/...
        politics/...
        sport/...
        tech/...
The folder path can be changed in the sidebar.
"""

import os
import nltk
import streamlit as st

# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------
DEFAULT_FOLDER = "bbc"
MAX_TOKENS_SHOWN = 5000     # long lists are cut for display (counts are always exact)
MAX_CHARS_SHOWN = 20000     # long texts are cut for display

LEVELS = ["One article", "One category", "Entire dataset"]

# Sentence boundaries: ? . ! and newline (the newline is handled in add_markers)
SEPARATORS = ["?", ".", "!"]

# The regular expression used by RegexpTokenizer. Alternatives, in order:
#   1) abbreviations : (?:[A-Za-z]\.){2,}      ->  u.s.t.h.b.   (at least 2 "letter + dot")
#   2) words/numbers : [£$€]?\w+(?:[@-]\w+|[.,]\d+)*[£$€%]?
#                      \w+            letters/digits         -> a2020, 2020a, hello
#                      [@-]\w+        @ or - then more       -> precision@10, state-of-the-art, 2020-45
#                      [.,]\d+        . or , then digits     -> 2020.45, 2020,45, precision@140.45
#                      [£$€]? ... [£$€%]?  optional currency  -> £347, 600€
#   3) punctuation   : [.?!]                    -> sentence separators
TOKEN_PATTERN = r"(?:[A-Za-z]\.){2,}|[£$€]?\w+(?:[@-]\w+|[.,]\d+)*[£$€%]?|[.?!]"

EXAMPLE_TEXT = (
    "This example provides a few guidelines to help you prepare your regular expressions. "
    "Abbreviations such as U.S.T.H.B., u.s.t.h.b., or u.S.T.h.b. should be properly handled. "
    "Certain metric names should also be recognized, for example: Precision@10, precision@140.45, "
    "Precision-453, or Precision-453.45. In addition, combinations of letters and digits like "
    "A2020, 2020A, £347 or 600€ should be taken into account. Parentheses are not considered in "
    "this context — for example: (DZ) or (AI). You should also handle numeric combinations such "
    "as 2020, 2020.45, 2020-45, or 2020,45, as well as mixed character sequences like "
    "State-of-the-art. Finally, remember that combinations of characters are perfectly fine — "
    "after all, most of this text is made up of them ;) and don’t forget to check the list of "
    "tokens extracted from this text."
)

# The three stemmers from NLTK
STEMMERS = {
    "Porter": nltk.PorterStemmer(),
    "Lancaster": nltk.LancasterStemmer(),
    "Snowball": nltk.SnowballStemmer("english"),
}


# ----------------------------------------------------------------------
# Part I - Loading and displaying text
# ----------------------------------------------------------------------
@st.cache_data
def load_dataset(folder):
    """Read the dataset into a dictionary: {category: [text of article 1, text of article 2, ...]}"""
    dataset = {}
    for category in sorted(os.listdir(folder)):
        category_path = os.path.join(folder, category)
        if not os.path.isdir(category_path):
            continue
        articles = []
        for file_name in sorted(os.listdir(category_path)):
            if file_name.endswith(".txt"):
                with open(os.path.join(category_path, file_name), "r", encoding="latin-1") as f:
                    articles.append(f.read())
        dataset[category] = articles
    return dataset


def get_text(dataset, level, category, article_number):
    """Return the text of one article, one category, or the entire dataset."""
    if level == "One article":
        return dataset[category][article_number - 1]
    if level == "One category":
        return "\n\n".join(dataset[category])
    all_articles = []
    for articles in dataset.values():
        all_articles.extend(articles)
    return "\n\n".join(all_articles)


# ----------------------------------------------------------------------
# Part II.A - Tokenization
# ----------------------------------------------------------------------
def tokenize(text):
    """Lowercase the text and split it into tokens with the regular expression."""
    tokenizer = nltk.RegexpTokenizer(TOKEN_PATTERN)
    return tokenizer.tokenize(text.lower())


def add_markers(text):
    """Tokenize and surround every sentence with <s> and </s>.
    A sentence ends at ?, ., ! or at the end of a line."""
    tokens = []
    in_sentence = False
    for line in text.split("\n"):                 # the newline is a separator
        for token in tokenize(line):
            if token in SEPARATORS:               # end of sentence
                if in_sentence:
                    tokens.append("</s>")
                    in_sentence = False
            else:
                if not in_sentence:               # start of a new sentence
                    tokens.append("<s>")
                    in_sentence = True
                tokens.append(token)
        if in_sentence:                           # end of line closes the sentence
            tokens.append("</s>")
            in_sentence = False
    return tokens


# ----------------------------------------------------------------------
# Part II.B - Stemming
# ----------------------------------------------------------------------
def stem_tokens(tokens, stemmer):
    """Stem every token (the <s> and </s> markers are kept unchanged)."""
    stemmed = []
    for token in tokens:
        if token in ("<s>", "</s>"):
            stemmed.append(token)
        else:
            stemmed.append(stemmer.stem(token))
    return stemmed


def get_vocabulary(tokens):
    """The set of unique tokens (sorted, to be easier to read)."""
    return sorted(set(tokens))


# ----------------------------------------------------------------------
# Display helpers
# ----------------------------------------------------------------------
def show_text_box(label, text):
    """Show a text in a scrollable box (cut if it is very long)."""
    if len(text) > MAX_CHARS_SHOWN:
        st.caption(f"{label}: showing the first {MAX_CHARS_SHOWN} of {len(text)} characters")
        text = text[:MAX_CHARS_SHOWN]
    st.text_area(label, text, height=300)


def show_list_box(label, items):
    """Show a list (one item per line) in a scrollable box (cut if it is very long)."""
    if len(items) > MAX_TOKENS_SHOWN:
        st.caption(f"{label}: showing the first {MAX_TOKENS_SHOWN} of {len(items)}")
        items = items[:MAX_TOKENS_SHOWN]
    st.text_area(label, "\n".join(items), height=300)


# ----------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------
st.set_page_config(page_title="TP NLP", layout="wide")

# ----- Sidebar -----
folder = st.sidebar.text_input("Dataset folder", DEFAULT_FOLDER)
if not os.path.isdir(folder):
    st.error(f"Folder '{folder}' not found. Put the BBC dataset there or change the path in the sidebar.")
    st.stop()

dataset = load_dataset(folder)
category = st.sidebar.selectbox("Select Category", list(dataset.keys()))
article_number = st.sidebar.number_input(
    "Number of article", min_value=1, max_value=len(dataset[category]), value=1
)
level = st.sidebar.radio("Data", LEVELS)

text = get_text(dataset, level, category, article_number)

# ----- Header -----
total_articles = sum(len(articles) for articles in dataset.values())
st.title("TP - Natural Language Processing (NLP)")
st.subheader("BBC News")
st.write(f"Contains **{total_articles} articles**, divided into **{len(dataset)} categories**: "
         + ", ".join(dataset.keys()))

# ----- Part 1: text display -----
if st.button("Display Text"):
    with st.expander("Part 1 — Text Display", expanded=True):
        show_text_box("Content", text)

# ----- Part 2: preprocessing controls -----
left, right = st.columns(2)
with left:
    tokenize_clicked = st.button("Apply Tokenization")
    use_example = st.checkbox("Tokenization example")
with right:
    normalize_clicked = st.button("Apply Normalization")
    stemmer_name = st.selectbox("Stemming algorithm", list(STEMMERS.keys()))

# ----- Part 2.A: tokenization -----
if tokenize_clicked:
    with st.expander("Part 2 — Text Preprocessing", expanded=True):
        if use_example:
            # Example from the TP: plain tokenization (no <s> </s>)
            tokens = tokenize(EXAMPLE_TEXT)
            col1, col2 = st.columns(2)
            with col1:
                show_text_box("Text", EXAMPLE_TEXT)
            with col2:
                st.text_area("Tokens", str(tokens), height=300)
        else:
            tokens = add_markers(text)
            vocabulary = get_vocabulary(tokens)
            col1, col2, col3 = st.columns(3)
            with col1:
                show_list_box("Tokens", tokens)
                st.write(f"**{len(tokens)}** Tokens")
            with col2:
                show_list_box("Vocabulary", vocabulary)
                st.write(f"**{len(vocabulary)}** Unique words")
            with col3:
                show_text_box("Text", text)

# ----- Part 2.B: stemming -----
if normalize_clicked:
    with st.expander("Part 2 — Text Preprocessing", expanded=True):
        tokens = add_markers(text)
        stemmed = stem_tokens(tokens, STEMMERS[stemmer_name])
        stemmed_vocabulary = get_vocabulary(stemmed)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            show_list_box("Tokens before stemming", tokens)
            st.write(f"**{len(tokens)}** Tokens")
        with col2:
            show_list_box(f"Tokens after {stemmer_name}", stemmed)
            st.write(f"**{len(stemmed)}** Tokens normalized")
        with col3:
            show_list_box("Vocabulary after stemming", stemmed_vocabulary)
            st.write(f"**{len(stemmed_vocabulary)}** Unique words")
        with col4:
            show_text_box("Text", text)
