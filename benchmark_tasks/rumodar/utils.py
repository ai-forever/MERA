from typing import Dict, List


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    completion = results[0]
    completion1 = str(completion).strip()
    acc = 0
    if len(doc["outputs"]) > 0:
        out = str(doc["outputs"])
        acc = int(completion1 == out)
    return {"acc": acc}
