import re
from pathlib import Path

import streamlit as st
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer


st.set_page_config(page_title="TP - Natural Language Processing (NLP)", layout="wide")

DATASET_DIR = Path(__file__).resolve().parent / "bbc"

EXAMPLE_TEXT = """This example provides a few guidelines to help you prepare your regular expressions. Abbreviations such as U.S.T.H.B., u.s.t.h.b., or u.S.T.h.b. should be properly handled. Certain metric names should also be recognized, for example: Precision@10, precision@140.45, Precision-453, or Precision-453.45. In addition, combinations of letters and digits like A2020, 2020A, £347 or 600€ should be taken into account. Parentheses are not considered in this context — for example: (DZ) or (AI). You should also handle numeric combinations such as 2020, 2020.45, 2020-45, or 2020,45, as well as mixed character sequences like State-of-the-art. Finally, remember that combinations of characters are perfectly fine — after all, most of this text is made up of them ;) and don't forget to check the list of tokens extracted from this text."""

# Commas are excluded, while other punctuation can still be tokenized.
TOKENIZER = re.compile(
    r"""(?ix)
    u\.s\.t\.h\.b\.                         # U.S.T.H.B.
    | precision[@-]\d+(?:\.\d+)?            # Precision@10, Precision-453.45
    | state-of-the-art                      # Fixed expression
    | £\d+(?:\.\d+)?                        # £347
    | \d+(?:\.\d+)?€                        # 600€
    | \w+(?:['’]\w+)?(?:[-@]\w+(?:['’]\w+)?)*  # Words and compounds
    | [^\w\s,]                              # Punctuation except commas
    """
)


@st.cache_data
def load_dataset(dataset_path):
    """Read the BBC text files, grouped by category."""
    dataset = {}
    root = Path(dataset_path)

    if not root.is_dir():
        return dataset

    for category_dir in sorted(root.iterdir()):
        if not category_dir.is_dir():
            continue

        articles = {}
        for file_path in sorted(category_dir.glob("*.txt")):
            try:
                articles[file_path.name] = file_path.read_text(
                    encoding="utf-8", errors="replace"
                )
            except OSError:
                pass

        if articles:
            dataset[category_dir.name] = articles

    return dataset


def tokenize_text(text, lowercase=True, sentence_markers=True):
    """Split on sentence boundaries, tokenize each sentence, and optionally mark it."""
    sentences = re.split(r"[?.!\n]+", text)
    tokens = []

    for sentence in sentences:
        sentence_tokens = TOKENIZER.findall(sentence.strip())

        if lowercase:
            sentence_tokens = [token.lower() for token in sentence_tokens]

        if sentence_tokens:
            if sentence_markers:
                tokens.append("<s>")
            tokens.extend(sentence_tokens)
            if sentence_markers:
                tokens.append("</s>")

    return tokens


def stem_tokens(tokens, stemmer):
    """Keep sentence markers unchanged and stem all other tokens."""
    return [
        token if token in ("<s>", "</s>") else stemmer.stem(token)
        for token in tokens
    ]


def show_token_statistics(tokens):
    vocabulary = set(tokens)
    st.markdown("#### Token statistics")
    st.write("**Total number of tokens:**", len(tokens))
    st.write("**Vocabulary:**")
    st.code(str(sorted(vocabulary)), language="text")
    st.write("**Number of unique tokens:**", len(vocabulary))


st.title("TP - Natural Language Processing (NLP)")
st.subheader("BBC News")

dataset = load_dataset(str(DATASET_DIR))

if not dataset:
    st.error(
        f"BBC dataset not found. Place the `bbc` folder next to this script:\n\n"
        f"`{DATASET_DIR}`"
    )
    st.stop()

categories = list(dataset.keys())
total_articles = sum(len(articles) for articles in dataset.values())

st.markdown(
    f"Contains **{total_articles} articles**, divided into "
    f"**{len(categories)} categories**: "
    + ", ".join(category.title() for category in categories)
    + " - published during 2004-2005."
)

# Sidebar controls
with st.sidebar:
    st.selectbox("Select Category", categories, key="category")
    category = st.session_state.category
    article_names = list(dataset[category].keys())

    st.number_input(
        "Number of article",
        min_value=1,
        max_value=max(1, len(article_names)),
        value=min(3, max(1, len(article_names))),
        step=1,
        key="article_number",
    )

    st.markdown("**Data**")
    level = st.radio(
        "Data level",
        ["One article", "One category", "Entire dataset"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**Tokenization options**")
    lowercase = st.checkbox("Convert tokens to lowercase", value=True)
    sentence_markers = st.checkbox("Add <s> and </s> sentence markers", value=True)

# Get the text at the selected level.
if level == "One article":
    selected_name = article_names[st.session_state.article_number - 1]
    selected_text = dataset[category][selected_name]
    text_description = f"{category.title()} — {selected_name}"
elif level == "One category":
    selected_name = None
    selected_text = "\n".join(dataset[category].values())
    text_description = f"Category: {category.title()} ({len(article_names)} articles)"
else:
    selected_name = None
    selected_text = "\n".join(
        article_text
        for category_articles in dataset.values()
        for article_text in category_articles.values()
    )
    text_description = f"Entire dataset ({total_articles} articles)"

if "token_results" not in st.session_state:
    st.session_state.token_results = None
if "stem_results" not in st.session_state:
    st.session_state.stem_results = None

# Part 1: display the selected text.
if st.button("Display Text"):
    st.session_state.display_text = True

with st.expander("Part 1 — Text Display", expanded=True):
    if st.session_state.get("display_text", False):
        st.caption(text_description)
        st.text_area("Content", value=selected_text, height=300, disabled=True)
    else:
        st.info("Click Display Text to show the selected article or dataset.")

st.divider()

# Part 2: tokenization and stemming controls.
left_button, right_button = st.columns(2)

with left_button:
    if st.button("Apply Tokenization", use_container_width=False):
        source_text = EXAMPLE_TEXT if st.session_state.get("tokenization_example", False) else selected_text
        tokens = tokenize_text(
            source_text,
            lowercase=lowercase,
            sentence_markers=sentence_markers,
        )
        st.session_state.token_results = {
            "text": source_text,
            "tokens": tokens,
            "description": "Tokenization example" if st.session_state.get("tokenization_example", False) else text_description,
        }
        st.session_state.stem_results = None

with right_button:
    algorithm = st.selectbox(
        "Stemming algorithm",
        ["Porter", "Lancaster", "Snowball", "All three"],
        label_visibility="visible",
    )
    if st.button("Apply Normalization", use_container_width=False):
        source_text = EXAMPLE_TEXT if st.session_state.get("tokenization_example", False) else selected_text
        tokens = tokenize_text(
            source_text,
            lowercase=lowercase,
            sentence_markers=sentence_markers,
        )
        stemmers = {
            "Porter": PorterStemmer(),
            "Lancaster": LancasterStemmer(),
            "Snowball": SnowballStemmer("english"),
        }
        choices = list(stemmers.items()) if algorithm == "All three" else [(algorithm, stemmers[algorithm])]
        st.session_state.token_results = {
            "text": source_text,
            "tokens": tokens,
            "description": "Tokenization example" if st.session_state.get("tokenization_example", False) else text_description,
        }
        st.session_state.stem_results = [
            (name, stemmer, stem_tokens(tokens, stemmer))
            for name, stemmer in choices
        ]

with st.container():
    st.checkbox("Tokenization example", value=False, key="tokenization_example")

with st.expander("Part 2 — Text Preprocessing", expanded=True):
    if st.session_state.token_results is None:
        st.info("Choose your options, then click Apply Tokenization or Apply Normalization.")
    else:
        result = st.session_state.token_results
        text_col, token_col = st.columns(2)

        with text_col:
            st.text_area("Text", value=result["text"], height=320, disabled=True)

        with token_col:
            st.text_area(
                "Tokens",
                value=str(result["tokens"]),
                height=320,
                disabled=True,
            )

        show_token_statistics(result["tokens"])

        if st.session_state.stem_results:
            for name, stemmer, stemmed_tokens in st.session_state.stem_results:
                st.markdown(f"### {name} Stemmer")
                before_col, after_col = st.columns(2)

                with before_col:
                    st.text_area(
                        "Tokens before stemming",
                        value=str(result["tokens"]),
                        height=220,
                        disabled=True,
                        key=f"{name}_before",
                    )
                    st.write("**Total before stemming:**", len(result["tokens"]))

                with after_col:
                    st.text_area(
                        "Tokens after stemming",
                        value=str(stemmed_tokens),
                        height=220,
                        disabled=True,
                        key=f"{name}_after",
                    )
                    st.write("**Total after stemming:**", len(stemmed_tokens))
                    st.write("**Vocabulary after stemming:**")
                    st.code(str(sorted(set(stemmed_tokens))), language="text")
                    st.write(
                        "**Number of unique tokens after stemming:**",
                        len(set(stemmed_tokens)),
                    )
