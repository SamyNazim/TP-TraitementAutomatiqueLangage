
import os
import re
import nltk
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
from wordcloud import WordCloud

# Dataset directory and initial example text
Dataset = os.path.join(os.path.dirname(__file__), "bbc")

# Tokenizer: recognizes words, numbers, abbreviations and special expressions.
# Commas and other punctuation are not extracted as tokens.
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
    | \d+(?:,\d{3})*(?:\.\d+)?(?:%|€|bn|m|k|am|pm|st|nd|rd|th)?(?![\w]|['’]\w)  # 100,000  2.5  10%  5bn  20th  4€
    | \w+(?:['’&]\w+)*(?:[-@]\w+(?:['’&]\w+)*)*            # don't, AT&T, R&B, e-mail, Kilroy-Silk
    """
)

# Choose the amount of text to process.
print("1. One article")
print("2. One category")
print("3. Entire dataset")

Level = input("Choose a level: ")

# Automatically find the category folders inside the BBC directory.
Categories = [
    folder for folder in os.listdir(Dataset)
    if os.path.isdir(os.path.join(Dataset, folder))
]

Text = ""

if Level == "1" or Level == "2":
    # Display the available categories.
    print("\nCategories:")
    for i, category in enumerate(Categories, 1):
        print(i, category)

    C = int(input("Choose a category: ")) - 1
    Category = Categories[C]
    Folder = os.path.join(Dataset, Category)
    Files = sorted(
        file for file in os.listdir(Folder)
        if file.endswith(".txt")
    )

    if Level == "1":
        # Display and select one article from the chosen category.
        for i, file in enumerate(Files, 1):
            print(i, file)

        A = int(input("Choose an article: ")) - 1

        with open(os.path.join(Folder, Files[A]), encoding="utf-8") as f:
            Text = f.read()

    else:
        # Read and combine all articles from the selected category.
        for file in Files:
            with open(os.path.join(Folder, file), encoding="utf-8") as f:
                Text += f.read() + "\n"

elif Level == "3":
    # Read every article from every category.
    for Category in Categories:
        Folder = os.path.join(Dataset, Category)

        for file in sorted(os.listdir(Folder)):
            if file.endswith(".txt"):
                with open(os.path.join(Folder, file), encoding="utf-8") as f:
                    Text += f.read() + "\n"

else:
    print("Invalid choice.")
    exit()


# Tokenization options
# Lowercase conversion and sentence markers can be enabled independently.
Lowercase = input("Convert tokens to lowercase? (y/n): ").lower() == "y"
Markers = input("Add <s> and </s>? (y/n): ").lower() == "y"

# Split sentences at ?, ., ! and newline characters.
Sentences = re.split(r"[?.!\n]+", Text)
Tokens = []
SentenceList = []  # One list of tokens per sentence (used for sentence segmentation).

for Sentence in Sentences:
    # Extract tokens from the current sentence.
    SentenceTokens = ExpReg.tokenize(Sentence)

    # Convert each token to lowercase if requested.
    if Lowercase:
        SentenceTokens = [token.lower() for token in SentenceTokens]

    # Add sentence boundary markers when the sentence contains tokens.
    if SentenceTokens:
        if Markers:
            Tokens.append("<s>")

        Tokens.extend(SentenceTokens)

        if Markers:
            Tokens.append("</s>")

        if Markers:
            SentenceList.append(["<s>"] + SentenceTokens + ["</s>"])
        else:
            SentenceList.append(SentenceTokens)


# Display tokenization results.
print("\nTokens:")
print(Tokens)

print("\nTotal number of tokens:", len(Tokens))

# A set contains each distinct token only once.
Vocabulary = set(Tokens)

print("\nVocabulary:")
print(sorted(Vocabulary))

print("\nNumber of unique tokens:", len(Vocabulary))


# Stemming: reduce words to their stems using NLTK.
Choice = input(
    "\nApply stemming? (n = no, 1 = Porter, "
    "2 = Lancaster, 3 = Snowball, 4 = all): "
).lower()

# Associate each choice with its stemming algorithm.
Stemmers = {
    "1": ("Porter", PorterStemmer()),
    "2": ("Lancaster", LancasterStemmer()),
    "3": ("Snowball", SnowballStemmer("english"))
}

if Choice == "4":
    Selected = list(Stemmers.values())
elif Choice in Stemmers:
    Selected = [Stemmers[Choice]]
else:
    Selected = []

for Name, Stemmer in Selected:
    # Keep sentence markers unchanged and stem the other tokens.
    StemmedTokens = [
        token if token in ("<s>", "</s>") else Stemmer.stem(token)
        for token in Tokens
    ]

    # Display the results for the selected stemming algorithm.
    print("\n", Name, "Stemmer")
    print("Tokens before stemming:", Tokens)
    print("Tokens after stemming:", StemmedTokens)
    print("Total tokens before stemming:", len(Tokens))
    print("Total tokens after stemming:", len(StemmedTokens))
    print("Vocabulary after stemming:", sorted(set(StemmedTokens)))
    print("Number of unique tokens after stemming:", len(set(StemmedTokens)))


# Text visualization and sentence segmentation.
# Both can use the original text or the normalized (stemmed) text.
Normalized = input(
    "\nUse normalized text for the word cloud and sentences? (n = no, 1 = Porter, "
    "2 = Lancaster, 3 = Snowball, 4 = all): "
).lower()

if Normalized == "4":
    Variants = list(Stemmers.values())
elif Normalized in Stemmers:
    Variants = [Stemmers[Normalized]]
else:
    Variants = [("Original text", None)]


# Word cloud: the size of each word reflects its frequency.
if input("\nShow the word cloud? (y/n): ").lower() == "y":
    for Name, Stemmer in Variants:
        # Ignore the sentence markers and stem the other tokens if requested.
        Words = [
            Stemmer.stem(token) if Stemmer else token
            for token in Tokens
            if token not in ("<s>", "</s>")
        ]

        if not Words:
            print("No words available to build a word cloud.")
            break

        Cloud = WordCloud(
            width=800, height=400, background_color="white", collocations=False
        ).generate(" ".join(Words))

        plt.figure(Name)
        plt.imshow(Cloud.to_array(), interpolation="bilinear")
        plt.title(Name)
        plt.axis("off")

    # Display all the word clouds.
    plt.show()


# Sentence segmentation: display each sentence and count them.
if input("\nShow the sentences? (y/n): ").lower() == "y":
    for Name, Stemmer in Variants:
        print("\nSentences (" + Name + "):")

        for SentenceTokens in SentenceList:
            # Keep sentence markers unchanged and stem the other tokens.
            if Stemmer:
                SentenceTokens = [
                    token if token in ("<s>", "</s>") else Stemmer.stem(token)
                    for token in SentenceTokens
                ]

            print(" ".join(SentenceTokens))

    print("\nNumber of sentences:", len(SentenceList))
