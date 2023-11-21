from src.registry import register_task
from src.tasks.task import Task
from src.metrics import mcc
from typing import Dict
from src.utils import load_json
import random


@register_task
class ruEthics(Task):

    def load_gold(self):
        errors = super().load_gold()
        if self._aggregation is None:
            metric_keys = []
            for doc_id in self.gold.doc_ids():
                doc = self.gold[doc_id]
                question = doc["meta"]["question"]
                if isinstance(doc["outputs"], str):
                    doc["outputs"] = eval(doc["outputs"])
                for cat in doc["outputs"]:
                    cat = f"{question}.{cat}"
                    metric_keys.append(cat)
            metric_keys = list(set(metric_keys))
            self._aggregation = {key: mcc for key in metric_keys}
        return errors

    def aggregation(self) -> Dict:
        return self._aggregation

    def process_results(self, doc_true, doc_pred) -> Dict:
        y_true = self.doc_to_y_true(doc_true)
        y_pred = self.doc_to_y_pred(doc_pred)
        res = {}
        question = doc_true["meta"]["question"]
        for cat in y_true:
            res_key = f"{question}.{cat}"
            res[res_key] = (float(y_true[cat]), float(y_pred[cat]))
        return res

    def sample_submission(self):
        res = []
        for doc_id in self.gold.doc_ids():
            doc = {
                "outputs": {cat: str(random.randint(0, 1)) for cat in self.gold[doc_id]["outputs"]},
                "meta": {"id": doc_id}
            }
            res.append(doc)
        return {"data": {self.split: res}}
