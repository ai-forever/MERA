import datasets
import numpy as np

from lm_eval.filters.extraction import RegexFilter
from lm_eval.models.api_models import JsonChatStr


CONTEXT_PLACEHOLDER = "<CONTEXT_PLACEHOLDER>"
OUTPUT_TYPE = "loglikelihood"
FALLBACK = "-1"
RUTIE_END_QUESTION_ID = 499

REGEXP = RegexFilter(regex_pattern=r"(\b([0-9])\b)", group_select=0, fallback=FALLBACK)


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    # make sure the datasets is sorted in ascending order for each dialog
    return datasets.Dataset.from_list(
        sorted(
            dataset,
            key=lambda x: [int(x["meta"]["dialog_id"]), int(x["meta"]["question_id"])],
        )
    )


def replace_targets(string, max_num, storage):
    # for consistency only
    if max_num == 0:
        return string

    # string contains parts like RUTIE_TARGET_0, RUTIE_TARGET_1, so on
    template = "RUTIE_TARGET_{idx}"
    for idx in range(max_num):
        to_fill = template.format(idx=idx)
        # replace each part with corresponding answer from storage
        string = string.replace(to_fill, storage["answers"][to_fill])
    return string


def _update_request(storage, request):
    # sanity check, if req_id > 0 and storage is empty => something went wrong
    if len(storage) == 0 and request.doc["meta"]["question_id"] != 0:
        print("No previous responses logged in storage!")
        return request

    max_num = request.doc["meta"]["question_id"]

    # when string passed (everywhere except for API calls)
    if isinstance(request.arguments[0], str):
        max_num = request.doc["meta"]["question_id"]
        new_pair = (
            replace_targets(request.arguments[0], max_num, storage),
            request.arguments[1],
        )
        request.arguments = new_pair
    else:
        max_num = request.doc["meta"]["question_id"]
        req = request.arguments[0].prompt
        kwargs = request.arguments[1]

        new_req = replace_targets(req, max_num, storage)
        new_req = JsonChatStr(new_req)
        request.arguments = (new_req, kwargs)

    return request


def _update_storage(storage, request):
    req_id = request.doc["meta"]["question_id"]

    # check that the set is over to clear storage
    if not isinstance(request.arguments[1], dict):
        dialog_ends = (
            request.doc["meta"]["question_id"] == RUTIE_END_QUESTION_ID
            and len(storage["candidates"]) == 1
        )
    else:
        dialog_ends = request.doc["meta"]["question_id"] == RUTIE_END_QUESTION_ID

    # clear storage after rutie ends
    if dialog_ends:
        return {}

    # loglikelihood setup
    if not isinstance(request.arguments[1], dict):
        # update storage only after running 2 requests for the same sample
        storage.setdefault("candidates", []).extend([request.resps[0][0]])
        # need 2 probas to decide on the answer
        if len(storage["candidates"]) == 2:
            # decide on the answer
            result = ["1", "2"][np.argmax(storage["candidates"])]
            # get string that includes the context
            storage.setdefault("answers", {})[f"RUTIE_TARGET_{req_id}"] = result
            # discard candidates
            storage["candidates"] = []

    # generative setup
    else:
        # pick LM answer and truncate spaces
        storage["candidates"] = request.resps[0].strip()

        # apply filter to response to get digit out of LM answer
        string_answer = extract_string([storage["candidates"]])
        filtered_answer = REGEXP.apply([[string_answer]], None)

        # might not find pattern - replace with FALLBACK
        result = (
            FALLBACK
            if (not len(filtered_answer) or not filtered_answer[0])
            else filtered_answer[0][0]
        )

        # store LM filtered answer
        storage.setdefault("answers", {})[f"RUTIE_TARGET_{req_id}"] = result

    return storage


def extract_string(nested_list):
    for item in nested_list:
        if isinstance(item, list):
            # If the item is a list, call the function recursively
            result = extract_string(item)
            if result is not None:
                return result
        elif isinstance(item, str):
            # If the item is a string, return it
            return item
    return None
