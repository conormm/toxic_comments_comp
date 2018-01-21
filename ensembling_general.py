import pandas as pd
import numpy as np
import os

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

datafiles = [os.path.join(os.path.abspath("data/predictions_mine"), i) for i in os.listdir("data/predictions_mine")]

dat = []
for data in datafiles:

    d = pd.read_csv(data)
    dat.append(d)
print("donr")

ensemb = dat[0].copy()
ensemb[labels] = (dat[0][labels] + dat[1][labels]) / len(dat)

ensemb.to_csv("lstm_nblr_ens.csv", index=False)