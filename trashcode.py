

##### evertthing below is waste


train = pd.read_csv("data/raw/train.csv")
test = pd.read_csv("data/raw/test.csv")

test[0]
[i.orth_.lower() for i in test[0] if not (i.is_punct or i.is_stop)]





def extract_postokens(nlptexts):

    for doc in nlptexts:
        for nc in doc.noun_chunks:
            while len(nc) > 1 and nc[0].dep_ not in ("amod", "compound"):
                nc = nc[1:]
                if len(nc) > 1:
                    nc.merge(nc.root.tag_, nc.text, nc.root.ent_type_)

                for ent in doc.ents:
                    if len(ent) > 1:
                        ent.merge(ent.root.tag_, ent.text, ent.label_)
        token_strings = []
        for token in tokens:
            text = token.text.replace(' ', '_')
            tag = token.ent_type_ or token.pos_
            token_strings.append('%s|%s' % (text, tag))
        yield ' '.join(token_strings)

list(extract_postokens(test))

### this works
for sent in test:
    for nc in sent.noun_chunks:
        if len(nc) > 1 and nc[0].dep_ not in ("amod", "compound"):
            nc = nc[1:]
        if len(nc) > 1:
            start = sent.text.find(nc.text)
            sent.merge(start, start + len(nc.text))


[i for i in test[1]]

sent.merge(start, start + len(nc))

[i for i in sent]
[i for i in test[1]]

doc1 = test[1]
doc1 = nlp(doc1.text)
doc1.text.find("own advice")
doc1.merge(10, 10 + len("own advice"))
doc1
[i for i in doc1]
len("own advice")
doc1.text.find("own advice")
doc1.merge()


[i for i in merged[0]]
merged


[i for i in test[1]]
len(list(extract_tokens(test)))


del token_strings



sen = sense_tokens(test)
next(sen)


sen

for sent in test:
    for nc in sent.noun_chunks:
        if len(nc) > 1 and nc[0].dep_ not in ("amod", "compound"):
            nc = nc[1:]
        if len(nc) > 1:
            sent.merge(0, len(nc))
    token_strings = []
    for token in sent:
        text = token.text.replace(" ", "_")
        tag = token.ent_type_ or token.pos_
        token_strings.append("{}|{}".format(text, tag))
    print(token_strings)

for sent in test:
    for token in sent:
        print(token, token.pos_)


doc = nlp(u'Los Angeles start.')
doc.merge(0, len('Los Angeles'))


[i for i in doc]



[i for i in test[4]]


for sent in test:

            res = str.replace(nc.text, " ", "|")
    print("".join(res))

dir(test[0][0])



for sent in test:
    for ent in sent.ents:
        print(ent, ent.label_)





list(extract_tokens(test))




def transform_texts(texts):
    # Stream texts through the models. We accumulate a buffer and release
    # the GIL around the parser, for efficient multi-threading.
    for doc in nlp.pipe(texts):
        # Iterate over base NPs, e.g. "all their good ideas"
        for np in doc.noun_chunks:
            # Only keep adjectives and nouns, e.g. "good ideas"
            while len(np) > 1 and np[0].dep_ not in ('amod', 'compound'):
                np = np[1:]
            if len(np) > 1:
                # Merge the tokens, e.g. good_ideas
                np.merge(np.root.tag_, np.text, np.root.ent_type_)
            # Iterate over named entities
            for ent in doc.ents:
                if len(ent) > 1:
                    # Merge them into single tokens
                    ent.merge(ent.root.tag_, ent.text, ent.label_)
        token_strings = []
        for token in tokens:
            text = token.text.replace(' ', '_')
            tag = token.ent_type_ or token.pos_
            token_strings.append('%s|%s' % (text, tag))
        yield ' '.join(token_strings)

list(transform_texts(test))

dir(spacy)



targets = train.columns[2:8]

#train = init_data_setup(train, labels=targets)
#test = init_data_setup(test, labels=targets, train=False)

df =

all_text = pd.concat([train.comment_text, test.comment_text], axis=1)
all_text


train.sample(100).comment_text.tolist()

files = ["hereis_something.csv", "hereis_something2.csv"]


[file[7:file.find(".")] for file in files]


files[0][4:7]


x = np.random.randn(10)

def ReLU(v):
    return 0 if v < 0 else v



np.where(x < 0, 0, x)

def log_loss(yhat, y):

    return -np.where(y == 1, np.log(yhat), 1-np.log(yhat))

yhat = np.array([0.8, 0.9, 0.1, 0.2, 0.95])
y = np.array([1, 1, 0, 0, 1])
yhat2 = np.array([1, 1, 0, 0, 1])
-log_loss(yhat2, y)

t1 = train.get_chunk()
t2 = train.get_chunk()
t2

def sense_tokens(text):

    for sent in text:
        for nc in sent.noun_chunks:
            if len(nc) > 1 and nc[0].dep_ not in ("amod", "compound"):
                nc = nc[1:]
            if len(nc) > 1:
                start = sent.text.find(nc.text)
                sent.merge(start, start + len(nc.text))


    for sent in text:
        token_strings = []
        for token in sent:
            if not (token.is_punct or token.is_space or token.is_stop):
                text = token.text.replace(" ", "_")
                tag = token.ent_type_ or token.pos_
                token_strings.append("{}|{}".format(text, tag))
        yield token_strings