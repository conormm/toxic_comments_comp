from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import read_data
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression
from scipy import sparse
from sklearn.cross_validation import StratifiedKFold, cross_val_score, LabelKFold
import re, string
from collections import defaultdict
import begin

re_tok = re.compile('([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s):
    return re_tok.sub(r' \1 ', s).split()

#np.random.seed(123)

# create an evaluation set
#eval_set = train.sample(2000)
#eval_ixs = eval_set.index

#eval_set[targets].sum(axis=0)

# drop evaluation set from code
#train.drop(eval_ixs, inplace=True)

vec_tfid = TfidfVectorizer(encoding="unicode",
                           tokenizer=tokenize,
                           ngram_range=(1, 3),
                           max_df=0.9, # if the word appears in 90% of cases ignore
                           min_df=3, # word must appear at least 3 times
                           sublinear_tf=True)

class NbLrClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, C=1.0, dual=False, n_jobs=1):
        self.C = C
        self.dual = dual
        self.n_jobs = n_jobs

    def predict(self, X):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict(X.multiply(self._r))

    def predict_proba(self, X):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict_proba(X.multiply(self._r))

    def fit(self, X, y):
        # Check that X and y have correct shape
        y = y
        x, y = check_X_y(X, y, accept_sparse=True)

        def pr(x, y_i, y):
            p = x[y==y_i].sum(axis=0)
            return (p+1) / ((y==y_i).sum()+1)

        self._r = sparse.csr_matrix(np.log(pr(X,1,y) / pr(X,0,y)))
        x_nb = X.multiply(self._r)
        self._clf = LogisticRegression(C=self.C, dual=self.dual, n_jobs=self.n_jobs).fit(x_nb, y)

        return self

@begin.start
def main():
    clf_nblr = NbLrClassifier(dual=True)

    X = vec_tfid.fit_transform(train.comment_text.values)
    test_X = vec_tfid.transform(test.comment_text.values)
    print("done")

    d = defaultdict(float)
    for y in targets:
        d[y]

    predictions = {"id": test["id"].values}

    for label in targets:
        print("fitting {}".format(label))
        y = train[label].values
        clf_nblr.fit(X=X, y=y)
        ll_score = np.mean(cross_val_score(clf_nblr, X=X, y=y,
                                           scoring="neg_log_loss",
                                           cv=3))
        print("The mean log loss score for y == {} is {}".format(label, ll_score))
        d[label] = ll_score

        predictions[label] = clf_nblr.predict_proba(X=test_X)[:, 1]

    preds = pd.DataFrame.from_dict(predictions)
    preds = pd.concat([preds.id, preds.loc[:, targets]], 1)

    preds.to_csv("nblr_preds.csv", index=False)