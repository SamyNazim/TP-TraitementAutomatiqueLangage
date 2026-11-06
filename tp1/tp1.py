import os 
import nltk

Text = "This example provides a few guidelines to help you prepare your regular expressions. Abbreviations such as U.S.T.H.B., u.s.t.h.b., or u.S.T.h.b. should be properly handled. Certain metric names should also be recognized, for example: Precision@10, precision@140.45, Precision-453, or Precision-453.45. In addition, combinations of letters and digits like A2020, 2020A, £347 or 600€ should be taken into account. Parentheses are not considered in this context — for example: (DZ) or (AI). You should also handle numeric combinations such as 2020, 2020.45, 2020-45, or 2020,45, as well as mixed character sequences like State-of-the-art. Finally, remember that combinations of characters are perfectly fine — after all, most of this text is made up of them ;) and don’t forget to check the list of tokens extracted from this text."

print("Original text:")

print(Text)

ExpReg = nltk.RegexpTokenizer(r'(?:[A-Z]\.)+|\w+')
tokens = ExpReg.tokenize(Text)  

print("Tokens extracted from the text:")
for sent in tokens:
    print(sent)

print("\nTotal number of tokens:", len(tokens))

