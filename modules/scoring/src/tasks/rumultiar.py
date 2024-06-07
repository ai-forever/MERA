from src.registry import register_task
from src.tasks.task import Task
from src.metrics import mean
from typing import Dict
import random


@register_task
class ruMultiAr(Task):
    def aggregation(self) -> Dict:
        return {"acc": mean}

    def process_results(self, doc_true, doc_pred) -> Dict:
        y_true = self.doc_to_y_true(doc_true)
        y_pred = self.doc_to_y_pred(doc_pred)
        return {"acc": y_true == y_pred}

    def sample_submission(self):
        res = []
        for doc_id in self.gold.doc_ids():
            doc = {
                "outputs": str(int(random.randint(-(10**6), 10**6))),
                "meta": {"id": doc_id},
            }
            res.append(doc)
        return {"data": {self.split: res}}
