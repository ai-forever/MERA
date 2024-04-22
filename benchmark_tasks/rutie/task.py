"""
The Turing-test Interview Emulation (RuTie) dataset.

Turing-test Interview Emulation (RuTie) is a Russian-language test for simulation of the Turing test. The dataset imitates a coherent dialogue with the subject, where a set of questions on various topics is asked, and it is necessary to choose the most correct of two answer options for each question. The dataset checks the various categories, including string operations, world knowledge, lexic, ethics, math, and many more. The dialogue context and memory of the models is a special focus of the dataset.

Homepage: https://mera.a-ai.ru/
"""

from numpy import argmax

from benchmark_tasks.custom_instances import ContextInstance
from benchmark_tasks.custom_task import MERATask
from lm_eval.api.metrics import mean


class ruTiE(MERATask):
    VERSION = 0
    DATASET_NAME = "rutie"

    OUTPUT_TYPE = "loglikelihood"

    CONTEXT_BASED = True
    CONTEXT_PLACEHOLDER = "<CONTEXT_PLACEHOLDER>"

    DATASET_LIMIT = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = sorted(
                    list(self.dataset["train"]), key=lambda x: x["meta"]["question_id"]
                )
            return self._training_docs

    def test_docs(self):
        # no shuffling so that the requests are ordered
        # to pass the previous as context for current
        if self.has_test_docs():
            return sorted(
                list(self.dataset["test"]), key=lambda x: x["meta"]["question_id"]
            )

    def doc_to_text(self, doc):
        return (
            doc["instruction"]
            .format(
                context=self.CONTEXT_PLACEHOLDER,
                question=doc["inputs"]["question"],
                choice1=doc["inputs"]["choice1"],
                choice2=doc["inputs"]["choice2"],
            )
            .replace("\n\n", "\n")
            + "\n"
            + "Ответ:"
        )

    def doc_to_text_without_instruction(self, doc):
        inputs = doc["inputs"]
        return inputs.strip()

    def doc_to_target(self, doc):
        target = doc["outputs"]
        return " " + target

    def _update_request(self, storage, request):
        if not len(storage) and request.doc["meta"]["question_id"] != 0:
            print("No previous responses logged in storage!")
            return request
        if request.doc["meta"]["question_id"] == 0:
            # no update for first request in dialog
            update_ctx = ""
        else:
            update_ctx = storage["string"]

        new_pair = (
            request.arguments[0].replace(self.CONTEXT_PLACEHOLDER, update_ctx),
            request.arguments[1],
        )
        request.arguments = new_pair
        return request

    def _update_storage(self, storage, request):
        # check that the set is over to clear storage
        if self.DATASET_LIMIT is None:
            end = self._instances[-1].doc
            self.DATASET_LIMIT = [
                end["meta"]["dialog_id"],
                end["meta"]["question_id"],
            ]
        if (
            request.doc["meta"]["dialog_id"] == self.DATASET_LIMIT[0]
            and request.doc["meta"]["question_id"] == self.DATASET_LIMIT[1]
            and len(storage["candidates"]) == 1
        ):
            rutie_ends = True
        else:
            rutie_ends = False
        # clear storage after rutie ends
        if rutie_ends:
            return {}
        # update storage only after running 2 requests for the same sample
        storage.setdefault("candidates", []).extend([request.resps[0][0]])
        if len(storage["candidates"]) == 2:
            # decide on the answer
            res = ["1", "2"][argmax(storage["candidates"])]
            # get string that includes the context
            storage["string"] = storage.get("string", "")
            # update the previous context with the new one and answer
            storage[
                "string"
            ] += "\n{question}\n1. {choice1}\n2. {choice2}\nОтвет: {result}".format(
                question=request.doc["inputs"]["question"],
                choice1=request.doc["inputs"]["choice1"],
                choice2=request.doc["inputs"]["choice2"],
                result=res,
            )
            # remove the first \n as it is already in instruction
            if storage["string"].startswith("\n"):
                storage["string"] = storage["string"][1:]
            storage["candidates"] = []
        return storage

    def construct_requests(self, doc, ctx, **kwargs):
        ll_1 = ContextInstance(
            request_type=self.OUTPUT_TYPE,
            doc=doc,
            arguments=(ctx, " 1"),
            idx=0,
            requests_updater=self._update_request,
            storage_updater=self._update_storage,
            **kwargs,
        )
        ll_2 = ContextInstance(
            request_type=self.OUTPUT_TYPE,
            doc=doc,
            arguments=(ctx, " 2"),
            idx=1,
            requests_updater=self._update_request,
            storage_updater=self._update_storage,
            **kwargs,
        )

        return [ll_1, ll_2]

    def process_results(self, doc, results):
        if len(doc["outputs"]) > 0:
            results = [
                res[0] for res in results
            ]  # only retain loglikelihoods, discard is_greedy
            gold = {"1": 0, "2": 1}[doc["outputs"]]
            pred = argmax(results)
            return {"acc": pred == gold}
        return {"acc": 0.0}  # if no label provided (test answers are secret)

    def aggregation(self):
        return {"acc": mean}

    def higher_is_better(self):
        return {"acc": True}
