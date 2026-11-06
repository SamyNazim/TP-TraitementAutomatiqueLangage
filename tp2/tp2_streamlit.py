import re
from pathlib import Path

import nltk

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend (Qt5Agg from the TP does not work inside Streamlit)
import matplotlib.pyplot as plt
import streamlit as st
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
from wordcloud import WordCloud


st.set_page_config(page_title="TP - Natural Language Processing (NLP)", layout="wide")

DATASET_DIR = Path(__file__).resolve().parent / "bbc"

EXAMPLE_TEXT = """This example provides a few guidelines to help you prepare your regular expressions. Abbreviations such as U.S.T.H.B., u.s.t.h.b., or u.S.T.h.b. should be properly handled. Certain metric names should also be recognized, for example: Precision@10, precision@140.45, Precision-453, or Precision-453.45. In addition, combinations of letters and digits like A2020, 2020A, £347 or 600€ should be taken into account. Parentheses are not considered in this context — for example: (DZ) or (AI). You should also handle numeric combinations such as 2020, 2020.45, 2020-45, or 2020,45, as well as mixed character sequences like State-of-the-art. Finally, remember that combinations of characters are perfectly fine — after all, most of this text is made up of them ;) and don't forget to check the list of tokens extracted from this text."""

ExpReg = nltk.RegexpTokenizer(
    r"""(?ix)
    u\.s\.t\.h\.b\.                                        # project-specific acronym
    | precision[@-]\d+(?:\.\d+)?                           # precision@10, precision-0.5
    | state-of-the-art
    | (?:https?://|www\.)[^\s<>"')]+[^\s<>"')\.,;:!?]      # URLs
    | (?:[a-z]\.){2,}                                      # U.K., i.e., e.g., A.D.
    | (?:mr|mrs|ms|dr|prof|inc|ltd)\.(?=\s+[A-Z0-9]|\s*$)  # Mr. Smith, Acme Inc.
    | [£$€]\d+(?:,\d{3})*(?:\.\d+)?(?:bn|m|k|million|billion|trillion)?(?![\w])  # £35bn, $1.3m, £100,000
    | \d+(?:\.\d+)?(?:-\w+)+                               # 6-3, 5-year-old, 2.5-3
    | \d+(?:,\d+)*(?:\.\d+)?(?:%|€|bn|m|k|am|pm|st|nd|rd|th)?(?![\w]|['’]\w)  # 100,000  2020,45  2.5  10%  5bn  20th  4€
    | \w+(?:['’&]\w+)*(?:[-@]\w+(?:['’&]\w+)*)*            # don't, AT&T, R&B, e-mail, Kilroy-Silk
    """
)
TOKENIZER = ExpReg

# Sentence separators: ?, !, newline, and a period that ends a sentence.
# A period inside a number ($1.13bn, 2020.45) or an abbreviation (U.S.T.H.B.) is not a separator.
SENTENCE_SPLITTER = re.compile(
    r"""[?!\n]+|(?<!\.[A-Za-z])(?<!\bMr)(?<!\bMrs)(?<!\bMs)(?<!\bDr)(?<!\bSt)(?<!\bInc)(?<!\bLtd)(?<!\bCo)(?<!\bvs)\.(?=["'’”)]*(?:\s|$))"""
)

STEMMERS = {
    "Porter": PorterStemmer(),
    "Lancaster": LancasterStemmer(),
    "Snowball": SnowballStemmer("english"),
}

_stem_cache = {}


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
                    encoding="latin-1"
                )
            except OSError:
                pass

        if articles:
            dataset[category_dir.name] = articles

    return dataset


MAX_DISPLAY_CHARS = 100_000  # Streamlit limits widget state size; full data is still used for all counts.


def preview(value, limit=MAX_DISPLAY_CHARS):
    """Truncate very long strings for display only."""
    value = str(value)
    if len(value) <= limit:
        return value
    return value[:limit] + f"\n... [display truncated: {len(value) - limit:,} more characters]"


def split_sentences(text, lowercase=True, sentence_markers=True):
    """Split on sentence boundaries and tokenize each sentence.

    Returns a list of sentences, each one being a list of tokens
    (optionally wrapped with <s> and </s>)."""
    sentences = []

    for sentence in SENTENCE_SPLITTER.split(text):
        sentence_tokens = TOKENIZER.tokenize(sentence.strip())

        if lowercase:
            sentence_tokens = [token.lower() for token in sentence_tokens]

        if sentence_tokens:
            if sentence_markers:
                sentence_tokens = ["<s>"] + sentence_tokens + ["</s>"]
            sentences.append(sentence_tokens)

    return sentences


def tokenize_text(text, lowercase=True, sentence_markers=True):
    """Tokenize the text sentence by sentence and return a flat list of tokens."""
    return [
        token
        for sentence in split_sentences(text, lowercase, sentence_markers)
        for token in sentence
    ]


def stem_tokens(tokens, stemmer):
    """Keep sentence markers unchanged and stem all other tokens."""
    stemmed = []
    for token in tokens:
        if token in ("<s>", "</s>"):
            stemmed.append(token)
            continue
        key = (type(stemmer).__name__, token)
        if key not in _stem_cache:
            _stem_cache[key] = stemmer.stem(token)
        stemmed.append(_stem_cache[key])
    return stemmed


def selected_stemmers(algorithm):
    if algorithm == "All three":
        return list(STEMMERS.items())
    return [(algorithm, STEMMERS[algorithm])]


def generate_wordcloud(tokens):
    """Build a word cloud image (numpy array) from a list of tokens."""
    words = [token for token in tokens if token not in ("<s>", "</s>")]
    if not words:
        return None
    wordcloud = WordCloud(
        width=800, height=400, background_color="white", collocations=False
    ).generate(" ".join(words))
    return wordcloud.to_array()


def show_token_statistics(tokens):
    vocabulary = set(tokens)
    st.markdown("#### Token statistics")
    st.write("**Total number of tokens:**", len(tokens))
    st.write("**Vocabulary:**")
    st.code(preview(sorted(vocabulary)), language="text")
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

    st.divider()
    st.markdown("**Normalization options**")
    algorithm = st.selectbox(
        "Stemming algorithm",
        ["Porter", "Lancaster", "Snowball", "All three"],
    )
    use_normalized = st.checkbox(
        "Use Normalized Text",
        value=False,
        help="Apply the selected stemmer before generating the word cloud or segmenting sentences.",
    )

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

# Text used by the Part 2 buttons (the example text or the selected data level).
use_example = st.session_state.get("tokenization_example", False)
source_text = EXAMPLE_TEXT if use_example else selected_text
source_description = "Tokenization example" if use_example else text_description

# Original text, or one normalized version per selected stemmer.
if use_normalized:
    variants = selected_stemmers(algorithm)
else:
    variants = [("Original text", None)]

for key in ("token_results", "stem_results", "wordcloud_results", "sentence_results"):
    if key not in st.session_state:
        st.session_state[key] = None

# Part 1: display the selected text.
if st.button("Display Text"):
    st.session_state.display_text = True

with st.expander("Part 1 — Text Display", expanded=True):
    if st.session_state.get("display_text", False):
        st.caption(text_description)
        st.text_area("Content", value=preview(selected_text), height=300, disabled=True)
    else:
        st.info("Click Display Text to show the selected article or dataset.")

st.divider()

# Part 2: tokenization, stemming, word cloud and sentence segmentation controls.
tokenize_col, normalize_col, cloud_col, sentence_col = st.columns(4)

with tokenize_col:
    if st.button("Apply Tokenization", use_container_width=False):
        tokens = tokenize_text(
            source_text,
            lowercase=lowercase,
            sentence_markers=sentence_markers,
        )
        st.session_state.token_results = {
            "text": source_text,
            "tokens": tokens,
            "description": source_description,
        }
        st.session_state.stem_results = None

with normalize_col:
    if st.button("Apply Normalization", use_container_width=False):
        tokens = tokenize_text(
            source_text,
            lowercase=lowercase,
            sentence_markers=sentence_markers,
        )
        st.session_state.token_results = {
            "text": source_text,
            "tokens": tokens,
            "description": source_description,
        }
        st.session_state.stem_results = [
            (name, stemmer, stem_tokens(tokens, stemmer))
            for name, stemmer in selected_stemmers(algorithm)
        ]

with cloud_col:
    if st.button("Word Cloud", use_container_width=False):
        with st.spinner("Generating word cloud..."):
            tokens = tokenize_text(
                source_text,
                lowercase=lowercase,
                sentence_markers=sentence_markers,
            )
            st.session_state.wordcloud_results = {
                "description": source_description,
                "normalized": use_normalized,
                "clouds": [
                    (
                        name,
                        generate_wordcloud(
                            stem_tokens(tokens, stemmer) if stemmer else tokens
                        ),
                    )
                    for name, stemmer in variants
                ],
            }

with sentence_col:
    if st.button("Sentence Segmentation", use_container_width=False):
        with st.spinner("Segmenting sentences..."):
            sentences = split_sentences(
                source_text,
                lowercase=lowercase,
                sentence_markers=sentence_markers,
            )
            st.session_state.sentence_results = {
                "text": source_text,
                "description": source_description,
                "count": len(sentences),
                "variants": [
                    (
                        name,
                        [
                            " ".join(
                                stem_tokens(sentence, stemmer) if stemmer else sentence
                            )
                            for sentence in sentences
                        ],
                    )
                    for name, stemmer in variants
                ],
            }

with st.container():
    st.checkbox("Tokenization example", value=False, key="tokenization_example")

with st.expander("Part 2 — Text Preprocessing", expanded=True):
    if (
        st.session_state.token_results is None
        and st.session_state.wordcloud_results is None
        and st.session_state.sentence_results is None
    ):
        st.info(
            "Choose your options, then click Apply Tokenization, Apply Normalization, "
            "Word Cloud or Sentence Segmentation."
        )

    if st.session_state.token_results is not None:
        result = st.session_state.token_results
        text_col, token_col = st.columns(2)

        with text_col:
            st.text_area("Text", value=preview(result["text"]), height=320, disabled=True)

        with token_col:
            st.text_area(
                "Tokens",
                value=preview(result["tokens"]),
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
                        value=preview(result["tokens"]),
                        height=220,
                        disabled=True,
                        key=f"{name}_before",
                    )
                    st.write("**Total before stemming:**", len(result["tokens"]))

                with after_col:
                    st.text_area(
                        "Tokens after stemming",
                        value=preview(stemmed_tokens),
                        height=220,
                        disabled=True,
                        key=f"{name}_after",
                    )
                    st.write("**Total after stemming:**", len(stemmed_tokens))
                    st.write("**Vocabulary after stemming:**")
                    st.code(preview(sorted(set(stemmed_tokens))), language="text")
                    st.write(
                        "**Number of unique tokens after stemming:**",
                        len(set(stemmed_tokens)),
                    )

    # Text visualization: word cloud.
    if st.session_state.wordcloud_results is not None:
        cloud_result = st.session_state.wordcloud_results
        st.markdown("### Word Cloud")
        st.caption(cloud_result["description"])

        for name, image in cloud_result["clouds"]:
            if cloud_result["normalized"]:
                st.markdown(f"#### {name} Stemmer")
            if image is None:
                st.warning("No words available to build a word cloud.")
                continue
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(image, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
            plt.close(fig)

    # Sentence segmentation.
    if st.session_state.sentence_results is not None:
        sentence_result = st.session_state.sentence_results
        st.markdown("### Sentence Segmentation")
        st.caption(sentence_result["description"])

        text_col, sentences_col = st.columns(2)

        with text_col:
            st.text_area(
                "Text",
                value=preview(sentence_result["text"]),
                height=320,
                disabled=True,
                key="sentences_source_text",
            )

        with sentences_col:
            for name, lines in sentence_result["variants"]:
                label = "Sentences" if len(sentence_result["variants"]) == 1 else f"Sentences — {name}"
                st.text_area(
                    label,
                    value=preview("\n".join(lines)),
                    height=320,
                    disabled=True,
                    key=f"sentences_{name}",
                )

        st.write("**Number of sentences:**", sentence_result["count"])