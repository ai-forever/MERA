import argparse
import json
import logging
import os
import pathlib

from lm_eval import evaluator, tasks, utils
from lm_eval.models import MODEL_REGISTRY

logging.getLogger("openai").setLevel(logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODEL_REGISTRY, help="Name of internal model class type.")
    parser.add_argument(
        "--model_args", default="", help="Comma separated string arguments for transformers model autoclass."
    )
    parser.add_argument(
        "--tasks", default=None, choices=utils.MultiChoice(tasks.ALL_TASKS), help="Comma separated list of task names."
    )
    parser.add_argument("--num_fewshot", type=int, default=0, help="Number of examples in few-shot context.")
    parser.add_argument("--batch_size", type=str, default=None, help="Batch size for model.")
    parser.add_argument(
        "--max_batch_size", type=int, default=None, help="Maximal batch size to try with --batch_size auto."
    )
    parser.add_argument("--device", type=str, default=None, help="PyTorch device string for running models.")
    parser.add_argument("--output_path", default=None, help="Path to store results of task run")
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Limit the number of examples per task. " "If <1, limit is a percentage of the total number of examples.",
    )
    parser.add_argument("--no_cache", action="store_true", help="Set to not cache model files.")
    parser.add_argument(
        "--decontamination_ngrams_path",
        default=None,
        help="Directory with the ngram files and info.json for decontamination",
    )
    parser.add_argument("--description_dict_path", default=None, help="Path to dictionary of custom task descriptions.")
    parser.add_argument(
        "--check_integrity",
        action="store_true",
        help="Whether to run the relevant part of the test suite for the tasks.",
    )
    parser.add_argument(
        "--write_out",
        action="store_true",
        default=False,
        help="Write details about prompts and logits to json for all tasks.",
    )
    parser.add_argument(
        "--output_base_path",
        type=str,
        default=None,
        help="Directory to which detailed eval info will be written. Defaults to present working dir.",
    )
    parser.add_argument(
        "--inference", action="store_true", default=False, help="Whether the procedure runs without labels."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.limit:
        print("WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT.")

    if args.tasks is None:
        task_names = tasks.ALL_TASKS
    else:
        task_names = utils.pattern_match(args.tasks.split(","), tasks.ALL_TASKS)

    print(f"Selected Tasks: {task_names}")

    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, "r") as f:
            description_dict = json.load(f)

    results = evaluator.simple_evaluate(
        model=args.model,
        model_args=args.model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        no_cache=args.no_cache,
        limit=args.limit,
        description_dict=description_dict,
        decontamination_ngrams_path=args.decontamination_ngrams_path,
        check_integrity=args.check_integrity,
        write_out=args.write_out,
        output_base_path=args.output_base_path,
        inference=args.inference,
    )

    if args.inference:
        output_base_path = (
            pathlib.Path(args.output_base_path) if args.output_base_path is not None else pathlib.Path(".")
        )
        with open(output_base_path.joinpath("evaluation_config.json"), "w", encoding="utf8") as file:
            json.dump(results["config"], file)

    dumped = json.dumps(results, indent=2)
    print(dumped)

    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w") as f:
            f.write(dumped)

    batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))
    print(
        f"{args.model} ({args.model_args}), limit: {args.limit}, provide_description: {args.provide_description}, "
        f"num_fewshot: {args.num_fewshot}, batch_size: {args.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
    )
    print(evaluator.make_table(results))


if __name__ == "__main__":
    main()
