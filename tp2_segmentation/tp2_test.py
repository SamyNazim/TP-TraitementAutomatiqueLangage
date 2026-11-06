
import os
import re
import nltk
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer

# Dataset directory and initial example text
Dataset = os.path.join(os.path.dirname(__file__), "bbc")

# Tokenizer: recognizes words, numbers, abbreviations and special expressions.
# Commas and other punctuation are not extracted as tokens.
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
    background_color='white'
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

# Split sentences at ?, ., ! and newline characters.
Sentences = re.split(r"[?.!\n]+", Text)
Tokens = []
SentenceList = []  # One list of tokens per sentence (used for sentence segmentation).

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
