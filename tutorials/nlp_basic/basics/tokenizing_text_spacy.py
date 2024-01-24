import spacy

nlp = spacy.load('en_core_web_sm')
text = "Mary, don’t slap the green witch"
print([str(token) for token in nlp(text.lower())])