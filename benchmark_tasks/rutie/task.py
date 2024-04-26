"""
The Turing-test Interview Emulation (RuTie) dataset.

Turing-test Interview Emulation (RuTie) is a Russian-language test for simulation of the Turing test. The dataset imitates a coherent dialogue with the subject, where a set of questions on various topics is asked, and it is necessary to choose the most correct of two answer options for each question. The dataset checks the various categories, including string operations, world knowledge, lexic, ethics, math, and many more. The dialogue context and memory of the models is a special focus of the dataset.

Homepage: https://mera.a-ai.ru/
"""

from numpy import argmax

from benchmark_tasks.custom_instances import ContextInstance
from benchmark_tasks.custom_task import MultipleChoiceMERATask


class ruTiE(MultipleChoiceMERATask):
    VERSION = 0
    DATASET_NAME = "rutie"

    CHOICES = ["1", "2"]

    CONTEXT_BASED = True
    CONTEXT_PLACEHOLDER = "<CONTEXT_PLACEHOLDER>"

    DATASET_LIMIT = None

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                # ensure the strict order of docs
                self._training_docs = sorted(
                    list(map(self.process_doc, self.dataset["train"])),
                    key=lambda x: [x["meta"]["dialog_id"], x["meta"]["question_id"]],
                )
            return self._training_docs

    def test_docs(self):
        # no shuffling so that the requests are ordered
        # to pass the previous as context for current
        if self.has_test_docs():
            return sorted(
                list(map(self.process_doc, self.dataset["test"])),
                key=lambda x: [x["meta"]["dialog_id"], x["meta"]["question_id"]],
            )

    def doc_to_text(self, doc):
        prompt = (
            doc["instruction"]
            .format(
                **doc["inputs"],
                context=self.CONTEXT_PLACEHOLDER,
            )
            .replace("\n\n", "\n")
            + "\nОтвет:"
        )
        return prompt.strip()

    def doc_to_text_without_instruction(self, doc):
        inputs = (
            "{context}\nВопрос: {question}\n1. {choice1}\n2. {choice2}".format(
                **doc["inputs"],
                context=self.CONTEXT_PLACEHOLDER,
            ).replace("\n\n", "\n")
            + "\nОтвет:"
        )
        return inputs.strip()

    def fewshot_examples(self, doc, k, rnd):
        docs = list(self.fewshot_docs())
        return docs[: k + 1]

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
        return [
            ContextInstance(
                request_type=self.OUTPUT_TYPE,
                doc=doc,
                arguments=(ctx, " {}".format(choice)),
                idx=i,
                requests_updater=self._update_request,
                storage_updater=self._update_storage,
                **kwargs,
            )
            for i, choice in enumerate(doc["choices"])
        ]
