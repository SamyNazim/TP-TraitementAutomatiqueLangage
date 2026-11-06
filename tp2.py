import os
import re
import nltk
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer

# Dataset directory and initial example text
Dataset = os.path.join(os.path.dirname(__file__), "bbc")

# Tokenizer: recognizes words, numbers, abbreviations and special expressions.
ExpReg = nltk.RegexpTokenizer(
    r"""(?ix)
    u\.s\.t\.h\.b\.
    | precision[@-]\d+(?:\.\d+)?
    | state-of-the-art
    | (?:https?://|www\.)[^\s<>"')]+[^\s<>"')\.,;:!?]
    | (?:[a-z]\.){2,}
    | (?:mr|mrs|ms|dr|prof|inc|ltd)\.(?=\s+[A-Z0-9]|\s*$)
    | [£$€]\d+(?:,\d{3})*(?:\.\d+)?(?:bn|m|k|million|billion|trillion)?(?![\w])
    | \d+(?:\.\d+)?(?:-\w+)+
    | \d+(?:,\d+)*(?:\.\d+)?(?:%|€|bn|m|k|am|pm|st|nd|rd|th)?(?![\w]|['’]\w)
    | \w+(?:['’&]\w+)*(?:[-@]\w+(?:['’&]\w+)*)*
    """
)

TOKENIZER = ExpReg

Text = "This example provides a few guidelines to help you prepare your regular expressions. Abbreviations such as U.S.T.H.B., u.s.t.h.b., or u.S.T.h.b. should be properly handled. Certain metric names should also be recognized, for example: Precision@10, precision@140.45, Precision-453, or Precision-453.45. In addition, combinations of letters and digits like A2020, 2020A, £347 or 600€ should be taken into account. Parentheses are not considered in this context — for example: (DZ) or (AI). You should also handle numeric combinations such as 2020, 2020.45, 2020-45, or 2020,45, as well as mixed character sequences like State-of-the-art. Finally, remember that combinations of characters are perfectly fine — after all, most of this text is made up of them ;) and don’t forget to check the list of tokens extractedfrom this text."

# TP2: WordCloud and sentence segmentation

# WordCloud generation
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Generate a word cloud from the original text
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white',
    collocations=False
).generate(Text)

# Display the word cloud on the figure
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')

# Save the image in the same directory as the script
plt.savefig(
    os.path.join(os.path.dirname(__file__), "wordcloud.png"),
    bbox_inches='tight',
    pad_inches=0
)

plt.close()

print("Word cloud saved as wordcloud.png")

# Sentence segmentation

# Sentence splitter
SENTENCE_SPLITTER = re.compile(
    r"""[?!\n]+|(?<!\.[A-Za-z])(?<!\bMr)(?<!\bMrs)(?<!\bMs)(?<!\bDr)(?<!\bSt)(?<!\bvs)(?:(?<!\w)\.|\.(?!\w))"""
)

# Sentence segmentation options
lowercase = input("\nConvert tokens to lowercase? (y/n): ").lower() == "y"
sentence_markers = input("Add <s> and </s> markers? (y/n): ").lower() == "y"

# Stemmers
STEMMERS = {
    "Porter": PorterStemmer(),
    "Lancaster": LancasterStemmer(),
    "Snowball": SnowballStemmer("english")
}

use_stemming = input("Apply stemming? (y/n): ").lower() == "y"

if use_stemming:
    print("\nSelect a stemmer:")
    print("1. Porter")
    print("2. Lancaster")
    print("3. Snowball")
    print("4. All three")

    Choice = input("Your choice: ").strip()

    if Choice == "1":
        Variants = [("Porter", STEMMERS["Porter"])]
    elif Choice == "2":
        Variants = [("Lancaster", STEMMERS["Lancaster"])]
    elif Choice == "3":
        Variants = [("Snowball", STEMMERS["Snowball"])]
    elif Choice == "4":
        Variants = list(STEMMERS.items())
    else:
        print("Invalid choice. Using the original tokens.")
        Variants = [("Original text", None)]
else:
    Variants = [("Original text", None)]

# Split sentences and tokenize them
SentenceList = []

for Sentence in SENTENCE_SPLITTER.split(Text):
    SentenceTokens = TOKENIZER.tokenize(Sentence.strip())

    if lowercase:
        SentenceTokens = [Token.lower() for Token in SentenceTokens]

    if SentenceTokens:
        if sentence_markers:
            SentenceTokens = ["<s>"] + SentenceTokens + ["</s>"]

        SentenceList.append(SentenceTokens)

# Display sentences and count them
if input("\nShow the sentences? (y/n): ").lower() == "y":
    for Name, Stemmer in Variants:
        print("\nSentences (" + Name + "):")

        for SentenceTokens in SentenceList:
            if Stemmer:
                SentenceTokens = [
                    Token if Token in ("<s>", "</s>") else Stemmer.stem(Token)
                    for Token in SentenceTokens
                ]

            print(" ".join(SentenceTokens))

    print("\nNumber of sentences:", len(SentenceList))