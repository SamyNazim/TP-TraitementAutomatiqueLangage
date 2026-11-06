import heapq
import random
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
# Interpolated N-gram language model: text completion
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


def generate_completion(model, context, lambdas, num_words, greedy, top_k, stop_at_end, seed):
    """Extend the context word by word using the interpolated model."""
    rng = random.Random(seed)
    context = list(context)
    words, steps = [], []
    first_candidates = None

    for step in range(1, num_words + 1):
        probabilities, usable, weights = next_word_probabilities(model, context, lambdas)
        ranked = heapq.nlargest(max(top_k, 10), probabilities.items(), key=lambda item: item[1])
        if not ranked:
            break

        if first_candidates is None:
            counts = model["counts"]
            rows = []
            for word, probability in ranked[:10]:
                row = {"Word": word}
                usable_by_order = {k: (h, c) for k, h, c in usable}
                for k in range(1, MAX_ORDER + 1):
                    if k in usable_by_order:
                        history, history_count = usable_by_order[k]
                        row[f"P {NGRAM_NAMES[k]}"] = counts[k].get(history + (word,), 0) / history_count
                    else:
                        row[f"P {NGRAM_NAMES[k]}"] = None
                row["Interpolated P"] = probability
                rows.append(row)
            first_candidates = pd.DataFrame(rows)

        if greedy:
            word, probability = ranked[0]
        else:
            pool = ranked[:top_k]
            word, probability = rng.choices(pool, weights=[p for _, p in pool], k=1)[0]

        steps.append(
            {
                "Step": step,
                "Context (last 3 tokens)": " ".join(context[-3:]),
                "Chosen word": word,
                "Probability": probability,
                "Highest order used": usable[-1][0],
            }
        )

        if word == "</s>":
            if stop_at_end:
                break
            context.extend(["</s>", "<s>"])
            continue

        words.append(word)
        context.append(word)

    return words, pd.DataFrame(steps), first_candidates


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
    st.markdown(
        "Type the beginning of a sentence and the model completes it. Next-word probabilities come from a "
        "**linear interpolation** of the unigram, bigram, trigram and quadrigram models trained on the "
        "data level selected in the sidebar:"
    )
    st.latex(
        r"P(w_n \mid w_{n-3}w_{n-2}w_{n-1}) = \lambda_4 P(w_n \mid w_{n-3}w_{n-2}w_{n-1}) "
        r"+ \lambda_3 P(w_n \mid w_{n-2}w_{n-1}) + \lambda_2 P(w_n \mid w_{n-1}) + \lambda_1 P(w_n)"
    )
    st.caption(
        f"Training data: {text_description}. Choose “Entire dataset” in the sidebar to use every article "
        "(better completions). Stemming is not applied here so that the completion stays readable; the "
        "lowercase option is respected, and sentence markers are always used."
    )

    lambda_columns = st.columns(4)
    lambdas = {}
    for column, k in zip(lambda_columns, range(1, MAX_ORDER + 1)):
        with column:
            lambdas[k] = st.slider(
                f"λ{k} ({NGRAM_NAMES[k]})",
                min_value=0.0,
                max_value=1.0,
                value=DEFAULT_LAMBDAS[k],
                step=0.05,
                key=f"lambda_{k}",
            )

    if sum(lambdas.values()) == 0:
        st.warning("All λ are 0: equal weights are used instead.")
        lambdas = {k: 1 / MAX_ORDER for k in lambdas}
    else:
        st.caption(
            f"The λ are normalised to sum to 1 (current sum: {sum(lambdas.values()):.2f}). "
            "If a history was never seen in the data, that order is skipped and the others are renormalised."
        )

    option_columns = st.columns(4)
    with option_columns[0]:
        num_words = st.slider("Words to generate", min_value=1, max_value=60, value=15)
    with option_columns[1]:
        selection_mode = st.radio(
            "Word selection",
            ["Most probable (greedy)", "Random sampling (top-K)"],
        )
    with option_columns[2]:
        top_k = st.slider(
            "Sampling pool K",
            min_value=2,
            max_value=50,
            value=10,
            disabled=selection_mode.startswith("Most"),
        )
    with option_columns[3]:
        stop_at_end = st.checkbox("Stop at end of sentence", value=True)

    with st.form("interpolation_form"):
        prompt = st.text_input(
            "Type the beginning of a sentence, then press Enter",
            placeholder="e.g. the government said that",
        )
        submitted = st.form_submit_button("Complete")

    if submitted:
        with st.spinner("Training the interpolated model on the selected data (cached afterwards)..."):
            cache_key = (
                level,
                category if level != "Entire dataset" else "",
                selected_name or "",
                lowercase,
            )
            interpolation_model = train_interpolation_model(cache_key, selected_text, lowercase)

        with st.spinner("Generating completion..."):
            context = prompt_to_context(prompt, lowercase)
            unknown_words = sorted(
                {token for token in context if token not in ("<s>", "</s>") and token not in interpolation_model["vocab_set"]}
            )
            words, steps_table, candidates_table = generate_completion(
                interpolation_model,
                context,
                lambdas,
                num_words,
                greedy=selection_mode.startswith("Most"),
                top_k=top_k,
                stop_at_end=stop_at_end,
                seed=None,
            )

        st.session_state.interpolation_results = {
            "prompt": prompt.strip(),
            "words": words,
            "steps": steps_table,
            "candidates": candidates_table,
            "unknown_words": unknown_words,
            "description": text_description,
            "tokens": interpolation_model["total"],
            "vocabulary": len(interpolation_model["vocab"]),
            "ngrams": {n: len(interpolation_model["counts"][n]) for n in range(1, MAX_ORDER + 1)},
        }

    interpolation_result = st.session_state.interpolation_results
    if interpolation_result is None:
        st.info("Type something above and click Complete (or press Enter).")
    else:
        st.markdown("#### Completion")
        completion = " ".join(interpolation_result["words"])
        if completion:
            st.markdown(
                f"{md_escape(interpolation_result['prompt'])} **{md_escape(completion)}**"
            )
        else:
            st.warning("The model could not generate any word for this text.")

        if interpolation_result["unknown_words"]:
            st.caption(
                "Words not found in the training data (the model falls back to shorter histories): "
                + ", ".join(interpolation_result["unknown_words"])
            )

        st.caption(
            f"Trained on {interpolation_result['description']} — "
            f"{interpolation_result['tokens']:,} tokens, {interpolation_result['vocabulary']:,} unique tokens; "
            + ", ".join(
                f"{count:,} unique {ngram_label(n).lower()}s"
                for n, count in interpolation_result["ngrams"].items()
            )
        )

        candidates_table = interpolation_result["candidates"]
        if candidates_table is not None:
            st.markdown("#### Top 10 candidates for the first generated word")
            probability_columns = [c for c in candidates_table.columns if c != "Word"]
            st.dataframe(
                candidates_table,
                hide_index=True,
                use_container_width=True,
                column_config={
                    column: st.column_config.NumberColumn(format="%.4f") for column in probability_columns
                },
            )

        steps_table = interpolation_result["steps"]
        if not steps_table.empty:
            st.markdown("#### Generation steps")
            st.dataframe(
                steps_table,
                hide_index=True,
                use_container_width=True,
                column_config={"Probability": st.column_config.NumberColumn(format="%.4f")},
            )