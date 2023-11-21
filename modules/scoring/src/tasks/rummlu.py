from src.registry import register_task
from src.tasks.task import Task
from src.metrics import mean
from typing import Dict


@register_task
class ruMMLU(Task):
    @property
    def choices(self):
        return ["A", "B", "C", "D"]

    def load_gold(self):
        errors = super().load_gold()
        if self._aggregation is None:
            criterias = []
            for doc_id in self.gold.doc_ids():
                doc = self.gold[doc_id]
                criterias.append(doc["meta"]["domain"])
            criterias = list(set(criterias))
            self._aggregation = {"acc": mean}
            for criteria in criterias:
                self._aggregation[f"acc.{criteria}"] = mean
        return errors

    def aggregation(self) -> Dict:
        return self._aggregation

    def process_results(self, doc_true, doc_pred) -> Dict:
        y_true = self.doc_to_y_true(doc_true)
        y_pred = self.doc_to_y_pred(doc_pred)
        res = y_true == y_pred
        criteria = doc_true["meta"]["domain"]
        return {"acc": res, f"acc.{criteria}": res}
