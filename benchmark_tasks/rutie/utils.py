import datasets
import numpy as np


CONTEXT_PLACEHOLDER = "<CONTEXT_PLACEHOLDER>"
OUTPUT_TYPE = "loglikelihood"
RUTIE_END_DIALOGUE_ID = 0
RUTIE_END_QUESTION_ID = 429


def _process_doc(doc: dict) -> dict:
    choices = ["1", "2"]
    if doc["outputs"]:
        gold = choices.index(doc["outputs"])
    else:
        gold = ""
    prompt = (
        doc["instruction"]
        .format(
            context=CONTEXT_PLACEHOLDER,
            question=doc["inputs"]["question"],
            choice1=doc["inputs"]["choice1"],
            choice2=doc["inputs"]["choice2"],
        )
        .replace("\n\n", "\n")
        + "\nОтвет:"
    )

    doc_to_text_without_instruction = (
        "{context}\n{question}\n1. {choice1}\n2. {choice2}".format(
            context=CONTEXT_PLACEHOLDER,
            question=doc["inputs"]["question"],
            choice1=doc["inputs"]["choice1"],
            choice2=doc["inputs"]["choice2"],
        ).replace("\n\n", "\n")
        + "\nОтвет:"
    )

    doc["doc_to_text_without_instruction"] = doc_to_text_without_instruction.strip()
    doc["choices"] = choices
    doc["gold"] = gold
    doc["query"] = prompt.strip()
    return doc


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    return datasets.Dataset.from_list(
        sorted(
            list(dataset.map(_process_doc)),
            key=lambda x: [int(x["meta"]["dialog_id"]), int(x["meta"]["question_id"])],
        )
    )


def _update_request(storage, request):
    if len(storage) == 0 and request.doc["meta"]["question_id"] != 0:
        print("No previous responses logged in storage!")
        return request
    if request.doc["meta"]["question_id"] == 0:
        # no update for first request in dialog
        update_ctx = ""
    else:
        update_ctx = storage["string"]

    new_pair = (
        request.arguments[0].replace(CONTEXT_PLACEHOLDER, update_ctx),
        request.arguments[1],
    )
    request.arguments = new_pair
    return request


def _update_storage(storage, request):
    default_rutie_end = (
        request.doc["meta"]["dialog_id"] == RUTIE_END_DIALOGUE_ID
        and request.doc["meta"]["question_id"] == RUTIE_END_QUESTION_ID
    )
    # loglikelihood setup
    if not isinstance(request.arguments[1], dict):
        # check that the set is over to clear storage
        rutie_ends = default_rutie_end and len(storage["candidates"]) == 1
        # clear storage after rutie ends
        if rutie_ends:
            return {}
        # update storage only after running 2 requests for the same sample
        storage.setdefault("candidates", []).extend([request.resps[0][0]])
        if len(storage["candidates"]) == 2:
            # decide on the answer
            res = ["1", "2"][np.argmax(storage["candidates"])]
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
    # generative setup
    else:
        # check that the set is over to clear storage
        # for gen task only one request per sample
        rutie_ends = default_rutie_end
        # clear storage after rutie ends
        if rutie_ends:
            return {}
        if request.resps[0].startswith(" "):
            # no need for leading space caused by no space in prompt end
            storage["candidates"] = request.resps[0][1:]
        else:
            storage["candidates"] = request.resps[0]
        storage["string"] = storage.get("string", "")
        storage[
            "string"
        ] += "\n{question}\n1. {choice1}\n2. {choice2}\nОтвет: {result}".format(
            question=request.doc["inputs"]["question"],
            choice1=request.doc["inputs"]["choice1"],
            choice2=request.doc["inputs"]["choice2"],
            result=storage["candidates"].strip(),
        )
        if storage["string"].startswith("\n"):
            storage["string"] = storage["string"][1:]
    return storage
