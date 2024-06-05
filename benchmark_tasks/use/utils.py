import re
from typing import Dict, List

import datasets
from numpy import mean


def _process_doc(doc: dict) -> dict:
    task_type = doc["meta"]["type"]
    if task_type in {"text", "multiple_choice_options_within_text"}:
        no_instruction = "Задание: {task}\n{text}\nОтвет:".format(**doc["inputs"])
    elif task_type == "matching":
        no_instruction = "Задание: {task}\nТекст: {text}\nРецензии: {additional_text}\nСписок терминов:\n{choices}\nОтвет:".format(
            **doc["inputs"]
        )
    elif task_type == "multiple_choice_based_on_text":
        no_instruction = "Задание: {task}\nТекст: {text}\nВарианты ответа:\n{choices}\nОтвет:".format(
            **doc["inputs"]
        )
    elif task_type == "multiple_choice_independent_options":
        no_instruction = "Задание: {task}\nВарианты ответа:\n{choices}\nОтвет:".format(
            **doc["inputs"]
        )
    doc["doc_to_text_without_instruction"] = no_instruction
    return doc


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.map(_process_doc)


def _process_results(doc, results):
    variant = doc["meta"]["variant"]
    if len(doc["outputs"]) > 0:
        id_task = doc["meta"]["id_task"]
        task_type = doc["meta"]["type"]
        answer = doc["outputs"]
        prediction = results[0]
        score = get_scores(task_type, id_task, answer, prediction)
        return {"grade_norm": (score, variant)}
    return {"grade_norm": (0.0, variant)}


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    processed_results = _process_results(doc, results)
    return processed_results


def multiple_choice_score(answer: str, prediction: str, is_task16=False) -> int:
    pred = prediction.split(",")
    ans = answer.split(",")
    if is_task16:
        while len(pred) < len(ans):
            pred.extend([-1])
        return max(
            0,
            len(set.intersection(set(ans), set(pred))) - len(pred) + len(ans),
        )
    ans_set = set(ans)
    pred_set = set(pred)
    return int(
        len(set.intersection(ans_set, pred_set)) == len(ans_set) == len(pred_set)
    )


def matching_score(answer: str, prediction: str) -> int:
    pred = prediction.split(",")
    ans = answer.split(",")
    score = 0
    if len(ans) != len(pred):
        return 0
    for idx, num in enumerate(ans):
        if num == pred[idx]:
            score += 1
    return score


def text_score(answer: str, prediction: str) -> int:
    pred = re.sub(r"[\d\W]+", "", prediction).lower()
    ans = answer.split(",")
    if pred in ans:
        return 1
    return 0


def get_scores(task_type, id_task, answer, prediction):
    if task_type == "matching":
        score = matching_score(answer, prediction)
    elif task_type == "text":
        score = text_score(answer, prediction)
    else:
        is_task16 = False
        if id_task == "16":
            is_task16 = True
        score = multiple_choice_score(answer, prediction, is_task16)
    return score


def overall_score(items, max_grade_point=34):
    overall_scores = {}
    for item in items:
        score, variant = item[0], item[1]
        if variant not in overall_scores:
            overall_scores[variant] = 0
        overall_scores[variant] += score

    average_overall_score = mean(
        [score / max_grade_point for score in overall_scores.values()]
    )
    return average_overall_score
