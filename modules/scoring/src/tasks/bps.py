from src.registry import register_task
from src.tasks.task import Task
from src.metrics import mean
from typing import Dict


@register_task
class BPS(Task):
    def aggregation(self) -> Dict:
        return {"acc": mean}

    def process_results(self, doc_true, doc_pred) -> Dict:
        y_true = self.doc_to_y_true(doc_true)
        y_pred = self.doc_to_y_pred(doc_pred)
        return {"acc": y_true == y_pred}
