import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict


def save_for_mera_submit(
    path: Path, task_name: str, samples: Dict[str, Any], serialized_results: str
):
    task_log_dir = path.joinpath(
        f"lm_harness_logs_{task_name}"
    )  # path to save logs of task
    task_log_dir.mkdir(exist_ok=False)  # create dir if not exist
    reverted_ans_queue: Dict[str, dict] = defaultdict(dict)
    reverted_docs: Dict[str, dict] = defaultdict(dict)
    for sample in samples[task_name]:
        idx = sample["doc"]["meta"]["id"]
        answers = [
            [resp_idx, resp] for resp_idx, resp in enumerate(sample["filtered_resps"])
        ]
        reverted_ans_queue[task_name][idx] = answers
        reverted_docs[task_name][idx] = sample["doc"]

    with open(
        task_log_dir.joinpath("output_answers.json"),
        "w",
        encoding="utf8",
    ) as file:
        json.dump(
            reverted_ans_queue[task_name],
            file,
            indent=4,
            ensure_ascii=False,
        )
    with open(task_log_dir.joinpath("input_docs.json"), "w", encoding="utf8") as file:
        json.dump(reverted_docs[task_name], file, indent=4, ensure_ascii=False)
    task_results_file = path.joinpath(f"{task_name}_result.json")
    with task_results_file.open("w", encoding="utf-8") as file:
        file.write(serialized_results)
