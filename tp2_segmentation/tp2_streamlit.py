import os
import re
import nltk
import streamlit as st
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
from wordcloud import WordCloud

# Dataset directory
Dataset = os.path.join(os.path.dirname(__file__), "bbc")

# Page configuration
st.set_page_config(
    page_title="Text Preprocessing",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Text Preprocessing")
st.write("Tokenization, stemming, word clouds and sentence segmentation.")

# Tokenizer: recognizes words, numbers and special expressions.
ExpReg = nltk.RegexpTokenizer(
    r"""(?ix)
    u\.s\.t\.h\.b\.
    | precision[@-]\d+(?:\.\d+)?
    | state-of-the-art
    | £\d+
    | \d+€
    | \w+(?:['’]\w+)?(?:[-@]\w+(?:['’]\w+)?)*
    | \d+(?:\.\d+)?
    """
)

# Available stemming algorithms
Stemmers = {
    "Porter": PorterStemmer(),
    "Lancaster": LancasterStemmer(),
    "Snowball": SnowballStemmer("english")
}

# Sidebar: choose the text to process
st.sidebar.header("1. Text Selection")

Level = st.sidebar.radio(
    "Choose a level:",
    ["One article", "One category", "Entire dataset"]
)

Text = ""
Categories = []
Files = []

if not os.path.isdir(Dataset):
    st.error(f"Dataset folder not found: {Dataset}")
    st.stop()

# Automatically find the category folders.
Categories = sorted(
    folder for folder in os.listdir(Dataset)
    if os.path.isdir(os.path.join(Dataset, folder))
)

if not Categories:
    st.error("No category folders were found in the BBC dataset.")
    st.stop()

if Level in ["One article", "One category"]:
    Category = st.sidebar.selectbox(
        "Choose a category:",
        Categories
    )

    Folder = os.path.join(Dataset, Category)

    Files = sorted(
        file for file in os.listdir(Folder)
        if file.endswith(".txt")
    )

    if not Files:
        st.error("No text articles were found in this category.")
        st.stop()

    if Level == "One article":
        Article = st.sidebar.selectbox(
            "Choose an article:",
            Files
        )

# Sidebar: tokenization options
st.sidebar.header("2. Tokenization")

Lowercase = st.sidebar.checkbox(
    "Convert tokens to lowercase",
    value=True
)

Markers = st.sidebar.checkbox(
    "Add sentence markers (<s>, </s>)",
    value=False
)

# Sidebar: stemming options
st.sidebar.header("3. Stemming")

StemChoice = st.sidebar.selectbox(
    "Choose stemming algorithm:",
    ["None", "Porter", "Lancaster", "Snowball", "All"]
)

# Sidebar: visualization options
st.sidebar.header("4. Visualization")

ShowCloud = st.sidebar.checkbox(
    "Show word clouds",
    value=True
)

ShowSentences = st.sidebar.checkbox(
    "Show sentences",
    value=True
)

NormalizedChoice = st.sidebar.multiselect(
    "Text versions for visualization:",
    ["Original text", "Porter", "Lancaster", "Snowball"],
    default=["Original text"]
)

Run = st.sidebar.button(
    "Process Text",
    type="primary",
    use_container_width=True
)

if Run:

    # Read the selected article, category or entire dataset.
    try:
        if Level == "One article":
            with open(
                os.path.join(Folder, Article),
                encoding="utf-8"
            ) as f:
                Text = f.read()

        elif Level == "One category":
            for file in Files:
                with open(
                    os.path.join(Folder, file),
                    encoding="utf-8"
                ) as f:
                    Text += f.read() + "\n"

        else:
            for Category in Categories:
                Folder = os.path.join(Dataset, Category)

                for file in sorted(os.listdir(Folder)):
                    if file.endswith(".txt"):
                        with open(
                            os.path.join(Folder, file),
                            encoding="utf-8"
                        ) as f:
                            Text += f.read() + "\n"

    except (OSError, UnicodeError) as e:
        st.error(f"Could not read the dataset: {e}")
        st.stop()

    if not Text.strip():
        st.warning("The selected text is empty.")
        st.stop()

    # Split the text into sentences.
    Sentences = re.split(r"[?.!\n]+", Text)

    Tokens = []
    SentenceList = []

    for Sentence in Sentences:
        SentenceTokens = ExpReg.tokenize(Sentence)

        if Lowercase:
            SentenceTokens = [
                token.lower() for token in SentenceTokens
            ]

        if SentenceTokens:
            if Markers:
                Tokens.append("<s>")

            Tokens.extend(SentenceTokens)

            if Markers:
                Tokens.append("</s>")
                SentenceList.append(
                    ["<s>"] + SentenceTokens + ["</s>"]
                )
            else:
                SentenceList.append(SentenceTokens)

    # Display basic information about the selected text.
    st.success("Text processed successfully!")

    st.subheader("📄 Dataset Information")

    col1, col2, col3 = st.columns(3)

    col1.metric("Characters", len(Text))
    col2.metric("Total tokens", len(Tokens))
    col3.metric("Number of sentences", len(SentenceList))

    st.write(f"**Selected level:** {Level}")

    if Level != "Entire dataset":
        st.write(f"**Selected category:** {Category}")

    if Level == "One article":
        st.write(f"**Selected article:** {Article}")

    # Display tokens and vocabulary.
    st.subheader("🔤 Tokenization Results")

    tab1, tab2 = st.tabs(["Tokens", "Vocabulary"])

    Vocabulary = sorted(set(Tokens))

    with tab1:
        st.write("**List of tokens:**")
        st.write(Tokens)

        st.write(f"**Total number of tokens:** {len(Tokens)}")

    with tab2:
        st.write("**Vocabulary:**")
        st.write(Vocabulary)

        st.write(
            f"**Number of unique tokens:** {len(Vocabulary)}"
        )

    # Apply the selected stemming algorithms.
    if StemChoice == "All":
        Selected = list(Stemmers.items())

    elif StemChoice in Stemmers:
        Selected = [(StemChoice, Stemmers[StemChoice])]

    else:
        Selected = []

    StemmedVersions = {}

    for Name, Stemmer in Selected:
        StemmedTokens = [
            token if token in ("<s>", "</s>")
            else Stemmer.stem(token)
            for token in Tokens
        ]

        StemmedVersions[Name] = StemmedTokens

    if Selected:
        st.subheader("🌱 Stemming Results")

        for Name, Stemmer in Selected:
            StemmedTokens = StemmedVersions[Name]
            StemmedVocabulary = sorted(set(StemmedTokens))

            with st.expander(f"{Name} Stemmer", expanded=True):

                col1, col2 = st.columns(2)

                col1.metric(
                    "Tokens before stemming",
                    len(Tokens)
                )

                col2.metric(
                    "Tokens after stemming",
                    len(StemmedTokens)
                )

                st.write("**Tokens before stemming:**")
                st.write(Tokens)

                st.write("**Tokens after stemming:**")
                st.write(StemmedTokens)

                st.write("**Vocabulary after stemming:**")
                st.write(StemmedVocabulary)

                st.write(
                    "**Unique tokens after stemming:**",
                    len(StemmedVocabulary)
                )

    # Prepare the text versions for visualization.
    Variants = []

    for Name in NormalizedChoice:
        if Name == "Original text":
            Variants.append((Name, None))

        elif Name in Stemmers:
            Variants.append((Name, Stemmers[Name]))

    # Word cloud visualization.
    if ShowCloud:
        st.subheader("☁️ Word Clouds")

        if not Variants:
            st.info(
                "Select at least one text version in the sidebar "
                "to display word clouds."
            )

        for Name, Stemmer in Variants:

            Words = [
                Stemmer.stem(token) if Stemmer else token
                for token in Tokens
                if token not in ("<s>", "</s>")
            ]

            if not Words:
                st.warning(
                    f"No words available for the {Name} word cloud."
                )
                continue

            Cloud = WordCloud(
                width=800,
                height=400,
                background_color="white",
                collocations=False
            ).generate(" ".join(Words))

            fig, ax = plt.subplots(figsize=(12, 6))

            ax.imshow(Cloud, interpolation="bilinear")
            ax.set_title(Name)
            ax.axis("off")

            st.pyplot(fig)
            plt.close(fig)

    # Sentence segmentation.
    if ShowSentences:
        st.subheader("📝 Sentence Segmentation")

        st.write(
            f"**Number of sentences:** {len(SentenceList)}"
        )

        if not Variants:
            st.info(
                "Select at least one text version in the sidebar "
                "to display sentences."
            )

        for Name, Stemmer in Variants:

            with st.expander(
                f"Sentences — {Name}",
                expanded=(Name == "Original text")
            ):

                for i, SentenceTokens in enumerate(
                    SentenceList, 1
                ):

                    if Stemmer:
                        DisplayTokens = [
                            token if token in ("<s>", "</s>")
                            else Stemmer.stem(token)
                            for token in SentenceTokens
                        ]
                    else:
                        DisplayTokens = SentenceTokens

                    st.write(
                        f"**Sentence {i}:** "
                        + " ".join(DisplayTokens)
                    )

    st.success("All selected operations are complete!")