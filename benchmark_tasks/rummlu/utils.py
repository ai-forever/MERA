from typing import Dict, List

from numpy import argmax


def process_results(doc: Dict, results: List[str]) -> Dict[str, float]:
    domain = doc["meta"]["domain"]
    has_outputs = len(doc["outputs"]) > 0
    is_generative = isinstance(results[0], str)
    if has_outputs and not is_generative:
        results = [res[0] for res in results]
        gold = ["A", "B", "C", "D"].index(doc["outputs"])
        pred = argmax(results)
        acc = float(pred == gold)
        return {"acc": acc, f"acc.{domain}": acc}
    if not has_outputs and not is_generative:
        return {
            "acc": 0.0,
            f"acc.{domain}": 0.0,
        }
    if has_outputs and is_generative:
        gold = doc["outputs"]
        completion = results[0]
        if not isinstance(gold, type(completion)):
            # cast gold to the same type as result
            gold = type(completion)(gold)
        acc = float(completion == gold)
        return {"em": acc, f"em.{domain}": acc}
    return {"em": 0.0, f"em.{domain}": 0.0}
