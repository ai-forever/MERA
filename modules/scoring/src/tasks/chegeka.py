from src.registry import register_task
from src.tasks.task import Task
from src.metrics import mean, metric_max_over_ground_truths
import transformers.data.metrics.squad_metrics as squad_metrics
from typing import Dict
import numpy as np


@register_task
class CheGeKa(Task):
    def aggregation(self) -> Dict:
        return {"f1": mean, "em": mean}

    def process_results(self, doc_true, doc_pred) -> Dict:
        y_true = self.doc_to_y_true(doc_true).split(";")
        y_pred = self.doc_to_y_pred(doc_pred)
        f1 = metric_max_over_ground_truths(squad_metrics.compute_f1, y_pred, y_true)
        em = metric_max_over_ground_truths(squad_metrics.compute_exact, y_pred, y_true)

        return {
            "f1": f1,
            "em": em,
        }

    def sample_submission(self):
        res = []
        for doc_id in self.gold.doc_ids():
            doc = {
                "outputs": " ".join(
                    np.random.choice(
                        self.gold[doc_id]["inputs"]["text"].split(), size=2
                    )
                ).lower(),
                "meta": {"id": doc_id},
            }
            res.append(doc)
        return {"data": {self.split: res}}
