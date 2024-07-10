import argparse
import glob
import json
import os
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional

import datasets
import numpy as np
from tqdm.auto import tqdm

from lm_eval.loggers.evaluation_tracker import GeneralConfigTracker
#MERA_FOLDER=/workspace/MERA/fix-mera/mera_run_030/test_mera_run
#MERA_MODEL_STRING="pretrained=/app/superllama,dtype=auto,max_length=16384"
BENCHMARK_STORAGE: Optional[str] = "ai-forever/MERA"
_TASKS = {}
GENERATIVE_SUFFIX = "_gen"
INPUT_DATE_FORMAT = "%Y-%m-%dT%H-%M-%S.%f"


def get_files_from_dir(dir_path):
    f = []
    for dir_path, dirn_ames, filenames in os.walk(dir_path):
        for fn in filenames:
            fn = os.path.join(dir_path, fn)
            f.extend([fn])
    return f


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(obj, file, ensure_ascii=False, indent=4)


def load_json(path):
    with open(path, encoding="utf-8") as file:
        text = json.loads(file.read().strip())
    return text


def load_jsonl(path):
    with open(path, encoding="utf-8") as file:
        result = [json.loads(line) for line in file.readlines()]
    return result


def extract_date(file_name: str) -> datetime:
    extract_str_date = file_name.split(".json")[0].split("_")[-1]
    date = datetime.strptime(extract_str_date, INPUT_DATE_FORMAT)
    return date


def register_task(cls):
    _TASKS[cls.__name__] = cls
    return cls


class BaseTask:
    @property
    def src_name(self):
        return self.__class__.__name__.lower()

    @property
    def dst_name(self):
        return self.__class__.__name__

    @property
    def outputs_path(self, index_to_get=0):
        if self.gen:
            filelist = glob.glob(
                os.path.join(
                    self.outputs_dir,
                    f"samples_{self.src_name}{GENERATIVE_SUFFIX}*.json",
                )
            )
            if len(filelist) == 0:
                # called only for originally generative tasks like multiq, chegeka, etc.
                filelist = glob.glob(
                    os.path.join(self.outputs_dir, f"samples_{self.src_name}*.json")
                )
        else:
            filelist = glob.glob(
                os.path.join(self.outputs_dir, f"samples_{self.src_name}*.json")
            )
            # filter out all tasks with GENERATIVE_SUFFIX as we don't need to process them
            filelist = list(filter(lambda x: GENERATIVE_SUFFIX not in x, filelist))
        if not filelist:
            # raise error if filelist is empty
            raise FileNotFoundError(
                "No samples to pack found, or there is an error in path processed"
            )
        # sorting filelist to get the latest
        filelist = sorted(filelist, key=extract_date, reverse=True)
        res = filelist[index_to_get]
        return res

    @property
    def submission_path(self):
        return os.path.join(self.dst_dir, f"{self.dst_name}.json")

    @staticmethod
    def doc_to_meta(doc):
        return doc["meta"]

    def doc_to_id(self, doc):
        return self.doc_to_meta(doc)["id"]

    def load(self):
        dataset = datasets.load_dataset(path=BENCHMARK_STORAGE, name=self.src_name)[
            "test"
        ]
        examples = dict()
        for example in dataset:
            doct_id = self.doc_to_id(example)
            examples[doct_id] = example
        return examples

    def __init__(
        self, outputs_dir, dst_dir, gen: bool, dataset_path: Optional[str] = None
    ):
        self.outputs_dir = outputs_dir
        self.dst_dir = dst_dir
        self.gen = gen
        self.dataset_path = dataset_path
        self.dataset = self.load()


class ClassificationTask(BaseTask):
    @property
    def choices(self):
        return ["0", "1"]

    def convert(self):
        submission = None
        try:
            submission = self.outputs_to_submission(load_jsonl(self.outputs_path))
            save_json(submission, self.submission_path)
        except FileNotFoundError:
            print(
                "No samples to pack found, or there is an error in path processed. Src:",
                self.src_name,
            )
        return submission

    def outputs_to_submission(self, outputs):
        res = []
        for doc in outputs:
            doc_id = int(self.doc_to_id(doc["doc"]))
            resp = doc["filtered_resps"]
            res.extend([self.doc_outputs_to_submission(doc_id, resp)])
        return {"data": {"test": res}}

    @staticmethod
    def parse_doc(doc):
        return doc[0]

    def doc_outputs_to_submission(self, doc_id, outputs):
        if self.gen:
            res = outputs[0]
        else:
            log_probs = np.zeros(len(outputs))
            for idx, doc in enumerate(outputs):
                prob = self.parse_doc(doc)
                log_probs[int(idx)] = prob
            idx = log_probs.argmax()
            res = self.choices[idx]
        return {
            "outputs": res,
            "meta": {"id": doc_id},
        }


class TextTask(ClassificationTask):
    def doc_outputs_to_submission(self, doc_id, outputs):
        res = {
            "outputs": outputs[0],
            "meta": {"id": doc_id},
        }
        return res


@register_task
class BPS(ClassificationTask):
    pass


@register_task
class LCS(ClassificationTask):
    @property
    def choices(self):
        return list(map(str, range(10)))


@register_task
class CheGeKa(TextTask):
    pass


@register_task
class MathLogicQA(ClassificationTask):
    @property
    def choices(self):
        return ["A", "B", "C", "D"]


@register_task
class MultiQ(TextTask):
    def doc_outputs_to_submission(self, doc_id, outputs):
        origin_doc = self.dataset[doc_id]
        text = outputs[0]
        pos = origin_doc["inputs"]["support_text"].find(text)
        if -1 == pos:
            pos = origin_doc["inputs"]["text"].find(text)
        res = {
            "outputs": [
                {
                    "label": origin_doc["outputs"][0]["label"],
                    "length": len(text),
                    "offset": pos,
                    "segment": text,
                },
            ],
            "meta": {
                "id": doc_id,
            },
        }
        return res


@register_task
class PARus(ClassificationTask):
    @property
    def choices(self):
        return ["1", "2"]


@register_task
class RCB(ClassificationTask):
    @property
    def choices(self):
        return ["1", "2", "3"]


@register_task
class ruDetox(TextTask):
    def doc_outputs_to_submission(self, doc_id, outputs):
        res = {
            "outputs": outputs[0][-1],
            "meta": {"id": doc_id},
        }
        return res


@register_task
class ruEthics(ClassificationTask):
    def doc_outputs_to_submission(self, doc_id, outputs):
        doc = super().doc_outputs_to_submission(doc_id, outputs)
        out = str(doc["outputs"])
        doc["outputs"] = {
            "virtue": out,
            "law": out,
            "moral": out,
            "justice": out,
            "utilitarianism": out,
        }
        return doc


@register_task
class ruHateSpeech(ClassificationTask):
    @property
    def choices(self):
        return ["1", "2"]


@register_task
class ruHHH(ClassificationTask):
    @property
    def choices(self):
        return ["1", "2"]


@register_task
class ruMMLU(ClassificationTask):
    @property
    def choices(self):
        return ["A", "B", "C", "D"]


@register_task
class ruModAr(TextTask):
    pass


@register_task
class ruMultiAr(TextTask):
    pass


@register_task
class SimpleAr(TextTask):
    pass


@register_task
class ruOpenBookQA(ClassificationTask):
    @property
    def choices(self):
        return ["A", "B", "C", "D"]


@register_task
class ruWorldTree(ClassificationTask):
    @property
    def choices(self):
        return ["A", "B", "C", "D"]


@register_task
class RWSD(ClassificationTask):
    @property
    def choices(self):
        return ["Да", "Нет"]


@register_task
class USE(TextTask):
    def doc_outputs_to_submission(self, doc_id, outputs):
        origin_doc = self.dataset[doc_id]
        res = {
            "outputs": outputs[0],
            "meta": {
                "id": doc_id,
                "id_task": origin_doc["meta"]["id_task"],
                "variant": origin_doc["meta"]["variant"],
            },
        }
        return res


@register_task
class ruTiE(TextTask):
    def load(self):
        dataset = datasets.load_dataset(path=BENCHMARK_STORAGE, name=self.src_name)[
            "test"
        ]
        dataset = [list(dataset)]
        return dataset

    @property
    def choices(self):
        return ["1", "2"]

    def outputs_to_submission(
        self, outputs: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, list]]:
        res_by_qid: Dict[int, Dict[int, Any]] = {}
        for doc in outputs:
            question_id = int(doc["doc"]["meta"]["question_id"])
            resp = doc["filtered_resps"]
            res_by_qid[question_id] = self.doc_outputs_to_submission(question_id, resp)
        res = []
        for dialog in self.dataset:
            new_dialog = []
            for question in dialog:
                dialog_id = question["meta"]["dialog_id"]
                question_id = question["meta"]["question_id"]
                question_id_outputs = res_by_qid.get(question_id)
                # check that question_id was passed to LM
                if question_id_outputs is not None:
                    new_question = {
                        "outputs": question_id_outputs['outputs'],
                        "meta": {
                            "dialog_id": dialog_id,
                            "question_id": question_id,
                        },
                    }
                    new_dialog.extend([new_question])
                else:
                    # if no, then stop, later questions should include answer for the current
                    # no current question means broken prompt for all others from this dialog
                    break
            res.extend([new_dialog])
        return {"data": {"test": res}}

    def doc_outputs_to_submission(self, doc_id, outputs):
        if self.gen:
            res = outputs[0]
        else:
            log_probs = np.zeros(len(outputs))
            for idx, doc in enumerate(outputs):
                prob = self.parse_doc(doc)
                log_probs[idx] = prob
            idx = log_probs.argmax()
            res = self.choices[idx]
        return {
            "outputs": res,
            "meta": {"id": doc_id},
        }


@register_task
class ruHumanEval(TextTask):

    def outputs_to_submission(self, outputs):
        res = []
        for doc in outputs:
            doc_id = int(self.doc_to_id(doc["doc"]))
            resp = doc["filtered_resps"][0]
            res.extend([self.doc_outputs_to_submission(doc_id, resp)])
        return {"data": {"test": res}}
    
    def doc_outputs_to_submission(self, doc_id, outputs):
        res = {
            "outputs": outputs,
            "meta": {
                "id": doc_id,
            },
        }
        return res


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_dir", type=str, help="lm-evaluation-harness outputs")
    parser.add_argument(
        "--dst_dir",
        type=str,
        default="submission/",
        help="dir to save files for submission",
    )
    parser.add_argument(
        "--gen",
        type=bool,
        default=False,
        help="generation or loglike setup.",
        action=argparse.BooleanOptionalAction,  # type: ignore[attr-defined]
    )
    parser.add_argument(
        "--model_args",
        type=str,
        default="",
        help="Comma separated string arguments for model, e.g. `pretrained=EleutherAI/pythia-160m,dtype=float32`",
    )
    res = parser.parse_known_args()[0]
    return res


def pack_submission_logs(outputs_dir: str, dst_dir: str):
    if os.path.isdir(outputs_dir):
        zip_dir = os.path.join(dst_dir, "logs_public")
        os.makedirs(zip_dir, exist_ok=True)
        files_to_pack = glob.glob(os.path.join(outputs_dir, "*.json"))
        for file_path in files_to_pack:
            shutil.copy2(file_path, zip_dir)
        zip_path = shutil.make_archive(zip_dir, "zip", zip_dir)
        shutil.rmtree(zip_dir)
        print("Logs to add with public submission stored at", zip_path)
    else:
        raise ValueError(f"{outputs_dir} is not directory")


def create_submission(outputs_dir, dst_dir, gen: bool):
    os.makedirs(dst_dir, exist_ok=True)
    for task_name, task_cls in tqdm(_TASKS.items(), total=len(_TASKS)):
        print("Process task", task_name)
        task = task_cls(outputs_dir=outputs_dir, dst_dir=dst_dir, gen=gen)
        _ = task.convert()
        print("---------------------")
    print("Packing logs for public submission...")
    pack_submission_logs(outputs_dir, dst_dir)
    zip_path = shutil.make_archive(dst_dir, "zip", dst_dir)
    print("Submission stored at", zip_path)


def preprocess_outputs_dir(outputs_dir: str, model_args: str) -> str:
    """
    User either provides "full" path to dir with jsons or provides path to
    folder of upper level and model_args to define subdir with jsons.
    If user explicitly provides model_args, parse it and use to define subdir.
    Otherwise, return the initial outputs_dir with no changes.
    """
    if model_args:
        # init predefined Tracker
        eval_tracker = GeneralConfigTracker()
        # init Tracker params and extract model_name_sanitized
        # model_source is not used yet,
        # do not need system_instruction and chat_template to find subdir name
        eval_tracker.log_experiment_args(
            model_args=model_args,
            model_source="",
            system_instruction="",
            chat_template="",
        )
        subdirectory = eval_tracker.model_name_sanitized
        # join paths
        full_path = os.path.join(outputs_dir, subdirectory)
        return full_path
    return outputs_dir


def main():
    args = get_args()
    outputs_dir = preprocess_outputs_dir(args.outputs_dir, args.model_args)
    create_submission(outputs_dir, args.dst_dir, gen=args.gen)


if __name__ == "__main__":
    main()
