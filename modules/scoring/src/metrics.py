import sklearn


def mean(arr):
    return sum(arr) / max(len(arr), 1)


def f1_macro_score(items):
    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    score = sklearn.metrics.f1_score(golds, preds, average="macro")
    return score


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """Compute max metric between prediction and each ground truth."""
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def mcc(items):
    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    score = sklearn.metrics.matthews_corrcoef(golds, preds)
    return score
