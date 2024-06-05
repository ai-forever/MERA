from typing import Dict, List

from transformers.data.metrics import squad_metrics

from lm_eval.api.metrics import metric_max_over_ground_truths


def _process_results_chegeka(doc, results):
    # - Pick the maximum likelihood prediction entity
    # - Evaluate the accuracy and token F1 PER EXAMPLE
    # - Average over all examples
    if len(doc["outputs"]) > 0:
        gold_label_set = doc["outputs"].split(";")
        pred = results[0]

        f1 = metric_max_over_ground_truths(
            squad_metrics.compute_f1, pred, gold_label_set
        )
        em = metric_max_over_ground_truths(
            squad_metrics.compute_exact, pred, gold_label_set
        )

        return {"f1": f1, "em": em}
    return {"f1": 0, "em": 0}  # if no label provided (test answers are secret)


def process_results_chegeka(doc: dict, results: List[str]) -> Dict[str, int]:
    processed_results = _process_results_chegeka(doc, results)
    return processed_results


def _process_results_multiq(doc, results):
    # - Pick the maximum likelihood prediction entity
    # - Evaluate the accuracy and token F1 PER EXAMPLE
    # - Average over all examples
    if len(doc["outputs"]) > 0:
        gold_label_set = [answer["segment"] for answer in doc["outputs"]]
        pred = results[0]

        f1 = metric_max_over_ground_truths(
            squad_metrics.compute_f1, pred, gold_label_set
        )
        em = metric_max_over_ground_truths(
            squad_metrics.compute_exact, pred, gold_label_set
        )

        return {"f1": f1, "em": em}
    return {"f1": 0, "em": 0}  # if no label provided (test answers are secret)


def process_results_multiq(doc: dict, results: List[str]) -> Dict[str, int]:
    processed_results = _process_results_multiq(doc, results)
    return processed_results
