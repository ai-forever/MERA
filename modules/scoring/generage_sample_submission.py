from src.worker import Worker
import shutil
import argparse


def generate_submissions(args):
    worker = Worker(conf=args.config_path, no_load_models=True)
    errors = worker.load()
    if len(errors):
        worker.log("Errors:")
        worker.log(errors)
    else:
        _, zip_path = worker.generate_random_baseline(
            args.submission_path[:-4], make_zip=True, produce_errors=args.produce_errors
        )
        shutil.rmtree(args.submission_path[:-4])
        worker.log(f"Submission stored at: {zip_path}")


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
        "--produce_errors", action="store_true", help="is delete prev deploy."
    )
    res = parser.parse_known_args()[0]
    return res


def main():
    args = get_args()
    generate_submissions(args)


if __name__ == "__main__":
    main()
