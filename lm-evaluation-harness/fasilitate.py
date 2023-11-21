import os
import json
import collections
import argparse
import pathlib
from typing import List

import lm_eval.tasks
import lm_eval.metrics
from lm_eval import evaluator

decontaminate_suffix = "_decontaminate"


def restore_records(dirs: List[pathlib.Path], base_path: pathlib.Path):
    print(dirs)
    process_res_queue = {}
    docs = {}
    for path in dirs:
        task_name = str(path)[len("lm_harness_logs_") :]
        with open(base_path.joinpath(path, "output_answers.json")) as resp_file, open(
            base_path.joinpath(path, "input_docs.json")
        ) as source_file:
            resps = json.load(resp_file)
            sources = json.load(source_file)
            for doc_id, resp in resps.items():
                process_res_queue[(task_name, doc_id)] = resp
                docs[(task_name, doc_id)] = sources[doc_id]

    return process_res_queue, docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_base_path", type=str, default=None)
    args = parser.parse_args()
    output_base_path = pathlib.Path(args.output_base_path) if args.output_base_path is not None else pathlib.Path(".")

    log_dirs = []
    task_names = []
    results = collections.defaultdict(dict)
    for path in os.listdir(output_base_path):
        if path.startswith("lm_harness_logs_"):
            log_dirs.append(path)
            task_names.append(str(path)[len("lm_harness_logs_") :])

    task_dict = lm_eval.tasks.get_task_dict(task_names)
    process_res_queue, docs = restore_records(log_dirs, output_base_path)
    decontaminate = False

    if output_base_path.joinpath("overlaps.json").is_file():
        decontaminate = True
        with open(output_base_path.joinpath("overlaps.json")) as file:
            overlaps = json.load(file)

    with open(output_base_path.joinpath("evaluation_config.json")) as file:
        config = json.load(file)

    vals = collections.defaultdict(list)

    # unpack results and sort back in order and return control to Task
    for (task_name, doc_id), requests in process_res_queue.items():
        requests.sort(key=lambda x: x[0])
        requests = [x[1] for x in requests]

        task = task_dict[task_name]
        doc = docs[(task_name, doc_id)]

        metrics = task.process_results(doc, requests)
        for metric, value in metrics.items():
            vals[(task_name, metric)].append(value)

            # Re-use the evaluation for the decontaminated set by just ignoring the overlaps
            if decontaminate and task_name in overlaps:
                if doc_id not in overlaps[task_name]:
                    vals[(task_name, metric + decontaminate_suffix)].append(value)

    # aggregate results
    for (task_name, metric), items in vals.items():
        task = task_dict[task_name]
        real_metric = metric  # key when looking up the metric with task.aggregation
        if metric.endswith(decontaminate_suffix):
            real_metric = metric.replace(decontaminate_suffix, "")  # decontaminated still uses the same metric
        results[task_name][metric] = task.aggregation()[real_metric](items)

        # hotfix: bleu, chrf, ter seem to be really expensive to bootstrap
        # so we run them less iterations. still looking for a cleaner way to do this

        stderr = lm_eval.metrics.stderr_for_metric(
            metric=task.aggregation()[real_metric],
            bootstrap_iters=min(config["bootstrap_iters"], 1000)
            if metric in ["bleu", "chrf", "ter"]
            else config["bootstrap_iters"],
        )

        if stderr is not None:
            results[task_name][metric + "_stderr"] = stderr(items)

    versions = collections.defaultdict(dict)
    for task_name, task in task_dict.items():
        versions[task_name] = task.VERSION

    results = {"results": dict(results), "config": config, "versions": dict(versions)}

    dumped = json.dumps(results, indent=2)
    print(dumped)

    print(evaluator.make_table(results))


if __name__ == "__main__":
    main()
