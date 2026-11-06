import os
import re
import nltk
from collections import Counter

# Dataset directory
Dataset = os.path.join(os.path.dirname(__file__), "bbc")

# Tokenizer: recognizes words, numbers, abbreviations and special expressions.
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
Lowercase = input("Convert tokens to lowercase? (y/n): ").lower() == "y"
Markers = input("Add <s> and </s>? (y/n): ").lower() == "y"

# Split sentences at ?, ., ! and newline characters.
Sentences = re.split(r"[?.!\n]+", Text)
Tokens = []

for Sentence in Sentences:
    # Extract tokens from the current sentence.
    SentenceTokens = ExpReg.tokenize(Sentence)

    # Convert tokens to lowercase if requested.
    if Lowercase:
        SentenceTokens = [token.lower() for token in SentenceTokens]

    # Add sentence boundary markers when the sentence contains tokens.
    if SentenceTokens:
        if Markers:
            Tokens.append("<s>")

        Tokens.extend(SentenceTokens)

        if Markers:
            Tokens.append("</s>")


# Display tokenization results.
print("\nTokens:")
print(Tokens)

print("\nTotal number of tokens:", len(Tokens))

# Display vocabulary.
Vocabulary = set(Tokens)

print("\nVocabulary:")
print(sorted(Vocabulary))

print("\nNumber of unique tokens:", len(Vocabulary))


# N-GRAM ANALYSIS

# 1. Choose N-gram size
print("\nChoose N-gram:")
print("1. Unigram\n2. Bigram\n3. Trigram\n4. Quadrigram\n5. Custom N-gram")
choice = input("Your choice: ")

if choice in ("1", "2", "3", "4"):
    n = int(choice)
elif choice == "5":
    n = int(input("Enter N: "))
else:
    n = 0
    print("Invalid choice.")

# 2. Process tokens if valid
if 0 < n <= len(Tokens):
    # Generate N-grams of length n
    ngrams = [tuple(Tokens[i:i + n]) for i in range(len(Tokens) - n + 1)]
    ngram_counts = Counter(ngrams)

    print(f"\nTotal number of {n}-grams: {len(ngrams)}")
    print(f"Number of unique {n}-grams: {len(ngram_counts)}")
    print(f"\nN-gram\t\tFrequency\tProbability")

    # Case 1: Unigram P(w_n) = C(w_n) / N
    if n == 1:
        total_tokens = len(Tokens)
        for gram, freq in ngram_counts.items():
            probability = freq / total_tokens
            print(f"{' '.join(gram)}\t\t{freq}\t\t{round(probability, 6)}")

    # Case 2: N-Gram (N > 1) P(w_n | history) = C(history + w_n) / C(history)
    else:
        history_len = n - 1
        history_counts = Counter(
            tuple(Tokens[i:i + history_len]) 
            for i in range(len(Tokens) - history_len + 1)
        )

        for gram, freq in ngram_counts.items():
            history = gram[:-1]
            probability = freq / history_counts[history]
            print(f"{' '.join(gram)}\t\t{freq}\t\t{round(probability, 6)}")

        # Optional Bigram Summary Statistics
        if n == 2:
            print("\nBigram Statistics:")
            print(f"Total bigrams (including repetitions): {len(ngrams)}")
            print(f"Unique bigrams: {len(ngram_counts)}")
            print(f"Repeated bigram occurrences: {len(ngrams) - len(ngram_counts)}")

else:
    print("Not enough tokens to generate this N-gram.")