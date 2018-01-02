import pandas as pd
import numpy as np
import os
import re, string
import begin

def init_data_setup(df, labels):

    df["no_label"] = 1-df[labels].max(axis=1)
    df["comment_text"] = df.comment_text.str.replace("\n", " ")
    df.comment_text.fillna("unknown", inplace=True)

    return df

def mk_comment_stats(comment_col):

    d = pd.DataFrame()

    d["length"] = comment_col.str.len()
    d["n_words"] = comment_col.str.split(" ").apply(lambda x: sum(1 for _ in x))
    d["n_qm"] = comment_col.str.count("'?'")
    d["n_exl"] = comment_col.str.count("'!")
    return d

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

targets = train.columns[2:8]

train = init_data_setup(train, labels=targets)
test = init_data_setup(test, labels=targets)


@begin.start
def main():

    folder = os.path.dirname(os.path.realpath(__file__))
    os.chdir(folder)
    print("I am in {} folder".format(folder))

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    targets = train.columns[2:8]
    print(targets)

    train = init_data_setup(train, labels=targets)
    test = init_data_setup(test)

    train.to_csv("train_mod1.csv", index=False)
    print("train saved")


