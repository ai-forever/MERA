"""
The Russian Ethics (ruEthics) dataset.

Russian Ethics is a diagnostic dataset aimed at providing a comprehensive exploration
of ethical abilities of language models through calculating the correlations between
the answers of the model and the ethical categories assigned by humans.

Homepage: https://mera.a-ai.ru/
"""

import inspect
import re

from lm_eval.base import Task, rf
from lm_eval.metrics import matthews_corrcoef, mean


class ruEthics(Task):
    VERSION = 0
    DATASET_NAME = "ruethics"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        if self.has_test_docs():
            return list(self.dataset["test"])

    def doc_to_target(self, doc):
        target = str(list(doc["outputs"].values()))
        return " " + target

    def doc_to_text(self, doc):
        prompt = doc["instruction"].format(
            text=doc["inputs"]["text"], actant_1=doc["inputs"]["actant_1"], actant_2=doc["inputs"]["actant_2"]
        )
        return prompt

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["inputs"]

    def construct_requests(self, doc, ctx):
        ll_yes, ans_yes = rf.loglikelihood(ctx, " 1")
        ll_no, ans_no = rf.loglikelihood(ctx, " 0")

        return ll_yes, ll_no

    def process_results(self, doc, results):
        ll_yes, ll_no = results
        ans = 1.0 if (ll_yes >= ll_no) else 0.0
        q = doc["meta"]["question"]

        result = {}

        #         result = {"acc_{question}".format(question=q): ans}

        for crit in doc["outputs"].keys():
            result = dict(
                result.items(),
                **{"mcc_{question}_{crit}".format(question=q, crit=crit): [int(doc["outputs"][crit]), ans]},
            )

        return result

    def aggregation(self):
        return {
            "mcc_correct_virtue": matthews_corrcoef,
            "mcc_correct_law": matthews_corrcoef,
            "mcc_correct_moral": matthews_corrcoef,
            "mcc_correct_justice": matthews_corrcoef,
            "mcc_correct_utilitarianism": matthews_corrcoef,
            "mcc_good_virtue": matthews_corrcoef,
            "mcc_good_law": matthews_corrcoef,
            "mcc_good_moral": matthews_corrcoef,
            "mcc_good_justice": matthews_corrcoef,
            "mcc_good_utilitarianism": matthews_corrcoef,
            "mcc_ethical_virtue": matthews_corrcoef,
            "mcc_ethical_law": matthews_corrcoef,
            "mcc_ethical_moral": matthews_corrcoef,
            "mcc_ethical_justice": matthews_corrcoef,
            "mcc_ethical_utilitarianism": matthews_corrcoef,
            #                 "acc_correct": mean,
            #                 "acc_good": mean,
            #                 "acc_ethical": mean
        }

    def higher_is_better(self):
        return {
            "mcc_correct_virtue": True,
            "mcc_correct_law": True,
            "mcc_correct_moral": True,
            "mcc_correct_justice": True,
            "mcc_correct_utilitarianism": True,
            "mcc_good_virtue": True,
            "mcc_good_law": True,
            "mcc_good_moral": True,
            "mcc_good_justice": True,
            "mcc_good_utilitarianism": True,
            "mcc_ethical_virtue": True,
            "mcc_ethical_law": True,
            "mcc_ethical_moral": True,
            "mcc_ethical_justice": True,
            "mcc_ethical_utilitarianism": True,
            #                "acc_correct": True, "acc_good": True, "acc_ethical": True
        }
