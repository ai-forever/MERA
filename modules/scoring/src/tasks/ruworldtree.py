from src.registry import register_task
from src.tasks.task import Task
from src.metrics import mean, f1_macro_score
from typing import Dict


@register_task
class ruWorldTree(Task):
    @property
    def choices(self):
        return ["A", "B", "C", "D"]

    def aggregation(self) -> Dict:
        return {"acc": mean, "f1_macro": f1_macro_score}

    def process_results(self, doc_true, doc_pred) -> Dict:
        y_true = self.doc_to_y_true(doc_true)
        y_pred = self.doc_to_y_pred(doc_pred)
        return {"acc": y_true == y_pred, "f1_macro": (y_true, y_pred)}
