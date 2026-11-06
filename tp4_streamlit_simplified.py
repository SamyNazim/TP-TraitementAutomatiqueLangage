import re
from collections import Counter
from pathlib import Path

import nltk

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend (Qt5Agg from the TP does not work inside Streamlit)
import matplotlib.pyplot as plt
import pandas as pd
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
    | \d+(?:[.,-]\d+)*(?:%|€|bn|m|k|am|pm|st|nd|rd|th)?(?![\w]|['’]\w)  # 2020  2020.45  2020-45  2020,45  6-3  10%  5bn  20th  4€
    | [a-z]+(?:\.[a-z]+)+(?![\w.])                        # dot.com, Plus.net
    | \w+(?:[-@]\w+)*                                      # e-mail, Kilroy-Silk, State-of-the-art (apostrophes split: don't -> don, t)
    """
)
TOKENIZER = ExpReg

# Sentence separators required by the TP: ?, ., ! and newline (\n).
# A period is a separator unless it is glued to a word character on BOTH sides
# (so $1.13bn, 2020.45, 325.5p, 3.817.69, dot.com and Plus.net stay inside their sentence),
# or it closes an acronym such as U.S.T.H.B. / i.e. / e.g., or a title (Mr., Dr., St., vs.).
# A period followed by a quote or bracket, by "..." or by no space at all still ends the sentence.
SENTENCE_SPLITTER = re.compile(
    r"""[?!\n]+|(?<!\.[A-Za-z])(?<!\bMr)(?<!\bMrs)(?<!\bMs)(?<!\bDr)(?<!\bSt)(?<!\bvs)(?:(?<!\w)\.|\.(?!\w))"""
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
                raw = file_path.read_bytes()
            except OSError:
                continue
            try:
                articles[file_path.name] = raw.decode("utf-8")
            except UnicodeDecodeError:
                # Original BBC files are Windows-1252 / latin-1 encoded.
                articles[file_path.name] = raw.decode("latin-1")

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


# ---------------------------------------------------------------------------
# N-gram language model (TP N°3)
# ---------------------------------------------------------------------------
NGRAM_NAMES = {1: "Unigram", 2: "Bigram", 3: "Trigram", 4: "Quadrigram"}
NGRAM_MODELS = ["Unigram", "Bigram", "Trigram", "Quadrigram"]


def ngram_label(n):
    return NGRAM_NAMES[n]


def count_ngrams(tokens, n):
    """Count every n-gram of the token list: tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)."""
    return Counter(zip(*(tokens[i:] for i in range(n))))


def build_ngram_model(tokens, n):
    """Frequencies and probabilities of the n-grams of a token list (formulas of the TP).

    Unigram : P(w_n)              = C(w_n) / N
    N-gram  : P(w_n | history)    = C(history + w_n) / C(history)   (bigram: C(w_{n-1} w_n) / C(w_{n-1}))
    """
    if not 0 < n <= len(tokens):
        return None

    ngrams_total = len(tokens) - n + 1
    ngram_counts = count_ngrams(tokens, n)

    if n == 1:
        history_counts = None
        total_tokens = len(tokens)  # N
    else:
        history_counts = count_ngrams(tokens, n - 1)

    rows = []
    # most_common() sorts by decreasing frequency (ties keep their order of first appearance).
    for gram, freq in ngram_counts.most_common():
        if n == 1:
            probability = freq / total_tokens
        else:
            probability = freq / history_counts[gram[:-1]]
        rows.append((" ".join(gram), freq, probability))

    return {
        "total": ngrams_total,
        "unique": len(ngram_counts),
        "rows": rows,
        "repeated": ngrams_total - len(ngram_counts),
    }


# ---------------------------------------------------------------------------
# Interpolated N-gram language model: text completion (Part 4)
# ---------------------------------------------------------------------------
MAX_ORDER = 4
DEFAULT_LAMBDAS = {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4}


@st.cache_resource(show_spinner=False)
def train_interpolation_model(cache_key, _text, lowercase):
    """Count the 1- to 4-grams of the whole selected data.

    `cache_key` identifies the data (the large `_text` is not hashed by Streamlit).
    Sentence markers are always added: they are needed to start and end a generated sentence.
    The history count C(w_{n-k+1..n-1}) of an order-k model is simply the count of the (k-1)-gram,
    so only the four n-gram counters need to be stored."""
    tokens = tokenize_text(_text, lowercase=lowercase, sentence_markers=True)
    counts = {n: count_ngrams(tokens, n) for n in range(1, MAX_ORDER + 1)}
    vocab = [gram[0] for gram in counts[1]]
    return {
        "counts": counts,
        "total": len(tokens),
        "vocab": vocab,
        "vocab_set": set(vocab),
    }


def order_contexts(model, context):
    """Orders usable for this context: (order k, history tuple, C(history)).

    Order 1 always works. Order k >= 2 needs a history of k-1 tokens that was seen in the data;
    otherwise it is skipped (the remaining weights are renormalised)."""
    usable = [(1, (), model["total"])]
    for k in range(2, MAX_ORDER + 1):
        if len(context) < k - 1:
            continue
        history = tuple(context[-(k - 1):])
        history_count = model["counts"][k - 1].get(history, 0)
        if history_count > 0:
            usable.append((k, history, history_count))
    return usable


def usable_weights(usable, lambdas):
    """Renormalise the lambdas over the usable orders so that they sum to 1."""
    total = sum(lambdas[k] for k, _, _ in usable)
    if total <= 0:
        return {k: 1 / len(usable) for k, _, _ in usable}
    return {k: lambdas[k] / total for k, _, _ in usable}


def next_word_probabilities(model, context, lambdas):
    """P(w | context) = sum_k lambda_k * C(h_k + w) / C(h_k) for every word w of the vocabulary."""
    usable = order_contexts(model, context)
    weights = usable_weights(usable, lambdas)
    counts = model["counts"]

    probabilities = {}
    for word in model["vocab"]:
        if word == "<s>":
            continue
        probability = 0.0
        for k, history, history_count in usable:
            freq = counts[k].get(history + (word,), 0)
            if freq:
                probability += weights[k] * freq / history_count
        if probability > 0:
            probabilities[word] = probability
    return probabilities, usable, weights


def prompt_to_context(prompt, lowercase):
    """Turn the typed text into a token context starting with <s>."""
    context = []
    for index, sentence in enumerate(split_sentences(prompt, lowercase, sentence_markers=False)):
        if index > 0:
            context.append("</s>")
        context.append("<s>")
        context.extend(sentence)
    if not context:
        context = ["<s>"]
    return context


COMPLETION_WORDS = 15  # maximum number of words added to the typed text


def generate_completion(model, context, lambdas=DEFAULT_LAMBDAS, num_words=COMPLETION_WORDS, only_order=None):
    """Extend the context with the most probable next word until </s>.

    With only_order=k, the pure k-gram model is used (it stops when its history was never seen);
    otherwise the interpolated model is used."""
    context = list(context)
    words = []
    for _ in range(num_words):
        probabilities, usable, _weights = next_word_probabilities(model, context, lambdas)
        if only_order and only_order not in [k for k, _, _ in usable]:
            break
        if not probabilities:
            break
        word = max(probabilities, key=probabilities.get)
        if word == "</s>":
            break
        words.append(word)
        context.append(word)
    return words


NUM_ALTERNATIVES = 3  # number of different completions shown for the bigram and trigram models


def generate_alternatives(model, context, k, num_alternatives=NUM_ALTERNATIVES):
    """Several completions of the pure k-gram model: one per most probable first word,
    each continued with the most probable next words."""
    pure_lambdas = {order: 1.0 if order == k else 0.0 for order in range(1, MAX_ORDER + 1)}
    probabilities, usable, _weights = next_word_probabilities(model, context, pure_lambdas)
    if k not in [order for order, _, _ in usable]:
        return []
    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    alternatives = []
    for first_word, _probability in ranked:
        if first_word == "</s>":
            continue
        rest = generate_completion(
            model, context + [first_word], pure_lambdas, COMPLETION_WORDS - 1, only_order=k
        )
        alternatives.append(" ".join([first_word] + rest))
        if len(alternatives) == num_alternatives:
            break
    return alternatives


def md_escape(text):
    return re.sub(r"([\\`*_$~])", r"\\\1", text)


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

for key in (
    "token_results",
    "stem_results",
    "wordcloud_results",
    "sentence_results",
    "ngram_results",
    "interpolation_results",
):
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

st.divider()

# Part 3: N-gram language model.
with st.expander("Part 3 — N-gram Language Model", expanded=True):
    model_col, generate_col, _ = st.columns([1, 1, 3], vertical_alignment="bottom")

    with model_col:
        ngram_model = st.selectbox("Select N-gram Model", NGRAM_MODELS, key="ngram_model")

    with generate_col:
        generate_ngrams = st.button("Generate ...", use_container_width=True)

    if generate_ngrams:
        with st.spinner("Generating N-grams..."):
            ngram_tokens = tokenize_text(
                source_text,
                lowercase=lowercase,
                sentence_markers=sentence_markers,
            )
            st.session_state.ngram_results = {
                "n": NGRAM_MODELS.index(ngram_model) + 1,
                "description": source_description,
                "normalized": use_normalized,
                "variants": [
                    (
                        name,
                        build_ngram_model(
                            stem_tokens(ngram_tokens, stemmer) if stemmer else ngram_tokens,
                            NGRAM_MODELS.index(ngram_model) + 1,
                        ),
                    )
                    for name, stemmer in variants
                ],
            }

    if st.session_state.ngram_results is None:
        st.info("Choose an N-gram model, then click Generate ...")
    else:
        ngram_result = st.session_state.ngram_results
        n = ngram_result["n"]
        label = ngram_label(n)
        st.caption(ngram_result["description"])

        for name, model in ngram_result["variants"]:
            if ngram_result["normalized"]:
                st.markdown(f"#### {name} Stemmer")

            if model is None:
                st.warning("Not enough tokens to generate this N-gram.")
                continue

            table = pd.DataFrame(
                model["rows"],
                columns=[label, "Frequency", "Probability P(Wn|W...)" if n > 1 else "Probability P(Wn)"],
            )
            st.dataframe(
                table,
                hide_index=True,
                use_container_width=True,
                height=320,
                column_config={table.columns[2]: st.column_config.NumberColumn(format="%.4f")},
            )
            st.markdown(
                f"**{model['total']}** {label}s with **{model['unique']}** unique {label}s"
            )

st.divider()

# Part 4: interpolation (text completion with an interpolated N-gram model).
with st.expander("Part 4 — Interpolation (N-gram Text Generation)", expanded=True):
    st.caption(
        "Type the beginning of a sentence and click Generate: it is completed by each n-gram model "
        "(bigram, trigram, quadrigram) and by their linear interpolation "
        f"(λ = {', '.join(str(DEFAULT_LAMBDAS[k]) for k in sorted(DEFAULT_LAMBDAS))}) "
        f"trained on: {text_description}."
    )

    with st.form("interpolation_form"):
        prompt = st.text_input("Text", placeholder="e.g. the government said that")
        submitted = st.form_submit_button("Generate")

    if submitted:
        with st.spinner("Generating..."):
            cache_key = (
                level,
                category if level != "Entire dataset" else "",
                selected_name or "",
                lowercase,
            )
            interpolation_model = train_interpolation_model(cache_key, selected_text, lowercase)
            context = prompt_to_context(prompt, lowercase)
            completions = []
            for k in (2, 3):
                completions.append((NGRAM_NAMES[k], generate_alternatives(interpolation_model, context, k)))
            pure_lambdas = {order: 1.0 if order == 4 else 0.0 for order in range(1, MAX_ORDER + 1)}
            quadrigram = generate_completion(interpolation_model, context, pure_lambdas, only_order=4)
            completions.append((NGRAM_NAMES[4], [" ".join(quadrigram)] if quadrigram else []))
            interpolation = generate_completion(interpolation_model, context)
            completions.append(("Interpolation", [" ".join(interpolation)] if interpolation else []))
        st.session_state.interpolation_results = {
            "prompt": prompt.strip(),
            "completions": completions,
        }

    interpolation_result = st.session_state.interpolation_results
    if interpolation_result is not None:
        prompt_text = md_escape(interpolation_result["prompt"])
        for label, completions in interpolation_result["completions"]:
            if not completions:
                st.markdown(f"**{label}:** _no completion (this history was never seen in the data)_")
            elif len(completions) == 1:
                st.markdown(f"**{label}:** {prompt_text} **{md_escape(completions[0])}**")
            else:
                st.markdown(f"**{label}:**")
                for completion in completions:
                    st.markdown(f"- {prompt_text} **{md_escape(completion)}**")