from src.registry import register_task
from src.tasks.task import Task
from src.metrics import mean, metric_max_over_ground_truths
from src.utils import load_json
import transformers.data.metrics.squad_metrics as squad_metrics
from typing import Dict
import numpy as np
import random


@register_task
class MultiQ(Task):

    def aggregation(self) -> Dict:
        return {"f1": mean, "em": mean}

    def process_results(self, doc_true, doc_pred) -> Dict:
        y_true = [answer["segment"] for answer in self.doc_to_y_true(doc_true)]
        y_pred = self.doc_to_y_pred(doc_pred)[0]["segment"]
        f1 = metric_max_over_ground_truths(squad_metrics.compute_f1, y_pred, y_true)
        em = metric_max_over_ground_truths(squad_metrics.compute_exact, y_pred, y_true)

        return {
            "f1": f1,
            "em": em,
        }

    def sample_submission(self):
        res = []
        for doc_id in self.gold.doc_ids():
            text = np.random.choice([self.gold[doc_id]["inputs"]["text"], self.gold[doc_id]["inputs"]["support_text"]])
            words = text.split()
            pos = np.random.choice(list(range(len(words))))
            word = words[pos: pos + random.randint(1, 3)]
            offset = text.find(word[0])
            segment = text[offset: text.find(word[-1]) + len(word[-1])]
            doc = {
                "outputs": [{
                    "label": "passage",
                    "length": len(segment),
                    "offset": offset,
                    "segment": segment}],
                "meta": {
                    "id": doc_id,
                    "bridge_answers": [
                        {
                            "label": "passage",
                            "offset": offset,
                            "length": len(segment),
                            "segment": segment
                        }
                    ]
                }
            }
            res.append(doc)
        return {"data": {self.split: res}}
    
    def remove_outputs(self):
        task = load_json(self.task_conf.origin_repo_path)
        for idx in range(len(task["data"]["test"])):
            task["data"]["test"][idx]["outputs"] = [{
                "label": "",
                "length": 0,
                "offset": 0,
                "segment": ""}]
            task["data"]["test"][idx]["meta"]["bridge_answers"] = [{
                "label": "",
                "length": 0,
                "offset": 0,
                "segment": ""
            }]
        return task
