"""
ruHumanEval
"""
import inspect

import ast
from .execute import check_correctness
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from lm_eval.base import Task, rf
from lm_eval.metrics import mean


class ruHumanEval(Task):
    VERSION = 0
    DATASET_NAME = "ruhumaneval"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = list(self.dataset["train"])
            return self._training_docs

    def test_docs(self):
        if self.has_test_docs():
            return list(self.dataset["test"])

    def doc_to_text(self, doc):
        prompt = doc["instruction"].format(**doc["inputs"])
        return prompt

    def doc_to_target(self, doc):
        return " " + str(doc["outputs"])

    def construct_requests(self, doc, ctx):
        return rf.greedy_until(ctx, {"until": ["."]})

    def execute_function(self, resp_func, doc, timeout=3.0, num_workers=2):
        test_cases = ast.literal_eval(doc["inputs"]["tests"])
        entry_point = doc["meta"]["entry_point"]

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            args = (resp_func, test_cases, entry_point, timeout)
            future = executor.submit(check_correctness, *args)
            result = future.result()
        return result["result"]

    def check_solution(self, true, pred):
        if not len(pred):
            return 0
        if len(true) != len(pred):
            return 0
        if type(true[0]) != type(pred[0]):
            return 0
        tests = []
        for idx, test in enumerate(true):
            try:
                if test == pred[idx]:
                    test.append(1)
                else:
                    test.append(0)
            except:
                test.append(0)
        return sum(tests) / len(tests)

    def compute_pass_k(self, n, c, k):
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    def process_results(self, doc, results, k):
        gold_label_set = doc["outputs"]
        arr = []
        for gen in range(len(results)):
            pred = results[gen]
            res = self.check_solution(gold_label_set, pred)
            if np.allclose([res], 1):
                arr.append(1)
            else:
                arr.append(0)
        correct = sum(arr)
        total = len(results)

        pass_5 = self.compute_pass_k(total, correct, k)
        pass_10 = self.compute_pass_k(total, correct, total)
        pass_1 = self.compute_pass_k(total, correct, 1)

        return {
            "pass@5": pass_5,
            "pass@10": pass_10,
            "pass@1": pass_1,
        }

    def higher_is_better(self):
        return {
            "pass@5": True,
            "pass@10": True,
            "pass@1": True,
        }

    def aggregation(self):
        return {
            "pass@5": mean,
            "pass@10": mean,
            "pass@1": mean,
        }
