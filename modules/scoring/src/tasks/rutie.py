from src.registry import register_task
from src.tasks.task import Task
from src.metrics import mean
from src.dataset import Dataset
from typing import Dict
import numpy as np
import datasets


@register_task
class ruTiE(Task):
    @property
    def choices(self):
        return ["1", "2"]

    def aggregation(self) -> Dict:
        return {"acc": mean}

    def process_results(self, doc_true, doc_pred) -> Dict:
        y_true = self.doc_to_y_true(doc_true)
        y_pred = self.doc_to_y_pred(doc_pred)
        return {"acc": mean([x == y for x, y in zip(y_true, y_pred)])}

    def doc_to_meta(self, doc):
        return doc[0]["meta"]

    def doc_to_id(self, doc):
        return self.doc_to_meta(doc)["dialog_id"]

    def doc_to_y_true(self, doc):
        return [x["outputs"] for x in doc]

    def doc_to_y_pred(self, doc):
        return [x["outputs"] for x in doc]

    def sample_submission(self):
        res = []
        for doc_id in self.gold.doc_ids():
            docs = []
            for origin_doc in self.gold[doc_id]:
                doc = {
                    "outputs": str(np.random.choice(self.choices)),
                    "meta": {
                        "dialog_id": origin_doc["meta"]["dialog_id"],
                        "question_id": origin_doc["meta"]["question_id"],
                    }
                }
                docs.append(doc)
            res.append(docs)
        return {"data": {self.split: res}}

    def load_gold(self):
        ds = datasets.load_dataset(path="ai-forever/MERA", name=self.name.lower())["test"]
        examples = dict()
        for example in [list(ds)]:
            doct_id = self.doc_to_id(example)
            examples[doct_id] = example
        self.gold = Dataset(
            local_path="",
            name=self.name,
            log=self.log,
            examples=examples
        )
        return []
