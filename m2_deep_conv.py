import pandas as pd
import spacy
import numpy as np
import os
from read_data import init_data_setup
from collections import Counter
import gensim

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

#df = init_data_setup(pd.read_csv("data/sample/sample_data.csv"),
#                     labels=labels)


train = init_data_setup(pd.read_csv("data/raw/train.csv"), labels=labels)
test = init_data_setup(pd.read_csv("data/raw/test.csv"), labels=None, train=False)

all_comments = pd.concat([train.comment_text, test.comment_text], axis=0)

all_comments.isnull().sum()
sample = all_comments.sample(5000)

nlp = spacy.load('en_core_web_sm')
nlp_comments = nlp.pipe(all_comments.values)

def sense_tokens(text):

    for sent in text:
        for nc in sent.noun_chunks:
            if len(nc) > 1 and nc[0].dep_ not in ("amod", "compound"):
                nc = nc[1:]
            if len(nc) > 1:
                # find the starting position of the nc in the doc
                start = sent.text.find(nc.text)
                sent.merge(start, start + len(nc.text))
        token_strings = []
        for token in sent:
            if not (token.is_punct or token.is_space or token.is_stop):
                text = token.text.lower().replace(" ", "_")
                tag = token.ent_type_ or token.pos_
                token_strings.append("{}|{}".format(text, tag))
        yield token_strings

# not using - using above instead

def token_to_int(text, lookup):

    f = lambda x: [lookup.get(i) for i in x]
    return map(f, text)

def extract_tokens(texts):

    tks = lambda x: [i for i in x]
    tokenised = map(tks, texts)
    return tokenised

sens_tokens = list(sense_tokens(nlp_comments))
print("donr")

#tokens = list(extract_tokens(sens_tokens))
flattokens = [i for j in sens_tokens for i in j]
token_counts = Counter(flattokens)
token_counts = {i : j for i, j in token_counts.items() if j >= 5}
token_lookup = {i : j for j, i in enumerate(flattokens, 1)}
[token_lookup.get(i) for i in sens_tokens[0]]

print("done")

word_vecs = gensim.models.Word2Vec(sens_tokens, min_count=5)
word_vecs.most_similar(positive=[ 'wise|ADJ'])

vector_loopup = np.zeros((len(token_lookup.keys()), 100))
for ix, word in enumerate(token_lookup.keys()):
    vector_loopup[ix, :] = word_vecs[word]

vector_loopup



sens_tokens

word_vecs[token_lookup[2]]
list(token_to_int(sens_tokens, token_lookup))

token_counts.most_common(100)

token_lookup.get(1)



##### evertthing below is waste
