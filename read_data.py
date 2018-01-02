import pandas as pd
import numpy as np
import os

print(os.path.dirname(os.path.realpath(__file__)))

train = pd.read_csv("data/train.csv")
train = pd.DataFrame(train)

targets = train.columns[2:8]
train["no_label"] = 1-train[targets].max(axis=1)

def init_data_setup(df):
    df["no_label"] = 1-df[targets].max(axis=1)
    df["comment_text"] = df.comment.text.str.replace("\n", " ")

    return df

def mk_comment_stats(comment_col):

    comment_col[""]






samp_text = train.comment_text.sample(100)
samp_text
pd.Series(samp_text).str.replace("\n", "")



