import numpy as np
import sklearn.metrics


def f1_score_multiclass_macro(items):
    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    fscore = sklearn.metrics.f1_score(golds, preds, average="macro")
    return np.max(fscore)
