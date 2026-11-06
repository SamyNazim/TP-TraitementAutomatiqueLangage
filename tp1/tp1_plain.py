"""
TP N°1 - Text Preprocessing (plain Python version, no Streamlit)

Run with:  python tp1_plain.py

Expected dataset layout (BBC News - Full Text):
    bbc/
        business/001.txt, 002.txt, ...
        entertainment/...
        politics/...
        sport/...
        tech/...
Change DATASET_FOLDER below if your folder has another name/location.
"""

import os
import nltk

# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------
DATASET_FOLDER = "bbc"
SHOW_LIMIT = 100        # max items/characters printed for long outputs (None = print everything)

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


def get_text(dataset, level, category=None, article_number=1):
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
# Printing helpers
# ----------------------------------------------------------------------
def shorten(items):
    """Keep only the first SHOW_LIMIT items (works for lists and strings)."""
    if SHOW_LIMIT is None or len(items) <= SHOW_LIMIT:
        return items
    return items[:SHOW_LIMIT]


def print_list(title, items):
    print(f"\n{title}")
    print(shorten(items))
    if SHOW_LIMIT is not None and len(items) > SHOW_LIMIT:
        print(f"... (showing the first {SHOW_LIMIT} of {len(items)})")


def choose(question, options):
    """Ask the user to pick one option from a numbered list."""
    print(f"\n{question}")
    for i, option in enumerate(options, start=1):
        print(f"  {i}. {option}")
    while True:
        answer = input("Your choice (number): ").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1]
        print("Invalid choice, try again.")


# ----------------------------------------------------------------------
# Main program
# ----------------------------------------------------------------------
def main():
    print("=== TP - Natural Language Processing (NLP) : BBC News ===")
    dataset = load_dataset(DATASET_FOLDER)
    total = sum(len(articles) for articles in dataset.values())
    print(f"Loaded {total} articles in {len(dataset)} categories: {', '.join(dataset)}")

    # --- Choose the level ---
    level = choose("Which data do you want to work on?", LEVELS)
    category = None
    article_number = 1
    if level != "Entire dataset":
        category = choose("Select a category:", list(dataset.keys()))
    if level == "One article":
        n = len(dataset[category])
        while True:
            answer = input(f"Article number (1-{n}): ").strip()
            if answer.isdigit() and 1 <= int(answer) <= n:
                article_number = int(answer)
                break
            print("Invalid number, try again.")

    text = get_text(dataset, level, category, article_number)

    # --- Part I: text display ---
    print("\n" + "=" * 60)
    print("PART 1 - TEXT DISPLAY")
    print("=" * 60)
    print(shorten(text))
    if SHOW_LIMIT is not None and len(text) > SHOW_LIMIT:
        print(f"... (showing the first {SHOW_LIMIT} of {len(text)} characters)")

    # --- Part II.A: tokenization ---
    print("\n" + "=" * 60)
    print("PART 2.A - TOKENIZATION")
    print("=" * 60)

    if input("\nRun the tokenization example from the TP? (y/n): ").strip().lower() == "y":
        print_list("Tokens of the example text:", tokenize(EXAMPLE_TEXT))

    tokens = add_markers(text)
    vocabulary = get_vocabulary(tokens)
    print_list("Tokens:", tokens)
    print(f"\nTotal number of tokens : {len(tokens)}")
    print_list("Vocabulary:", vocabulary)
    print(f"\nNumber of unique tokens: {len(vocabulary)}")

    # --- Part II.B: stemming (all three algorithms) ---
    print("\n" + "=" * 60)
    print("PART 2.B - STEMMING")
    print("=" * 60)
    for name, stemmer in STEMMERS.items():
        stemmed = stem_tokens(tokens, stemmer)
        stemmed_vocabulary = get_vocabulary(stemmed)
        print(f"\n----- {name} stemmer -----")
        print_list("Tokens before stemming:", tokens)
        print_list("Tokens after stemming:", stemmed)
        print(f"\nTokens before stemming : {len(tokens)}")
        print(f"Tokens after stemming  : {len(stemmed)}")
        print_list("Vocabulary after stemming:", stemmed_vocabulary)
        print(f"\nUnique tokens after stemming: {len(stemmed_vocabulary)}")


if __name__ == "__main__":
    main()
