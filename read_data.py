import pandas as pd
import numpy as np
import os
import re, string
import begin

def init_data_setup(df, labels, train=True):

    if train:
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
    d["prop_caps"] = comment_col.str.count("[A-Z]") / samp.str.len()
    return d

if __name__ == '__main__':
    def main():

        train = pd.read_csv("data/raw/train.csv")
        test = pd.read_csv("data/raw/test.csv")

        #%cd data/sample
        #%pwd
        train.sample(500).to_csv("sample_data.csv", index=False)

        targets = train.columns[2:8]

        train = init_data_setup(train, labels=targets)
        test = init_data_setup(test, labels=targets, train=False)
        print("done")


