from src.registry import register_task
from src.tasks.task import Task
from src.metrics import mean
from src.utils import load_json, random_choice
import numpy as np


@register_task
class ruHumanEval(Task):

    def aggregation(self):
        return {f"pass@{k}": mean for k in self.task_conf.ks}

    @staticmethod
    def check_solution(y_true, y_pred):
        if not len(y_pred):
            return 0
        if len(y_true) != len(y_pred):
            return 0
        if isinstance(y_true[0], type(y_pred[0])):
            return 0
        tests = []
        for idx, test in enumerate(y_true):
            try:
                if test == y_pred[idx]:
                    tests.append(1)
                else:
                    tests.append(0)
            except:
                tests.append(0)
        return sum(tests) / len(tests)

    @staticmethod
    def compute_pass_k(n, c, k):
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    def process_results(self, doc_true, doc_pred):
        gold_label_set = self.doc_to_y_true(doc_true)
        results = self.doc_to_y_pred(doc_pred)
        arr = []
        for gen in range(len(results)):
            y_pred = results[gen]
            res = self.check_solution(gold_label_set, y_pred)
            if np.allclose([res], 1):
                arr.append(1)
            else:
                arr.append(0)
        correct = sum(arr)
        total = len(results)

        return {f"pass@{k}": self.compute_pass_k(total, correct, k) for k in self.task_conf.ks}

    def sample_submission(self):
        res = []
        for doc_id in self.gold.doc_ids():
            tests = eval(self.gold[doc_id]["inputs"]["tests"])
            doc = {
                "outputs": [[
                    str(random_choice(list(x.values()))) for x in tests] for _ in range(max(self.task_conf.ks))],
                "meta": {"id": doc_id}
            }
            res.append(doc)
        return {"data": {self.split: res}}
