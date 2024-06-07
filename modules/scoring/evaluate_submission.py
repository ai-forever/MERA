from src.worker import Worker
from src.utils import save_json
import json
import argparse


def evaluate_submissions(args):
    worker = Worker(conf=args.config_path, no_load_models=False)
    errors = worker.load()
    if len(errors):
        worker.log(errors)
        worker.log("Evaluate with errors...")
    res = worker.evaluate(local_path=args.submission_path)
    save_json(res, args.results_path)
    worker.log(f"Submission stored at: {args.results_path}")
    worker.log(f"Evaluation result: {json.dumps(res, ensure_ascii=False, indent=4)}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/main.yaml",
        help="path to auth config",
    )
    parser.add_argument(
        "--submission_path",
        type=str,
        default="submission.zip",
        help="path to submission",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="submission_results.json",
        help="path to submission results",
    )
    res = parser.parse_known_args()[0]
    return res


def main():
    args = get_args()
    evaluate_submissions(args)


if __name__ == "__main__":
    main()
