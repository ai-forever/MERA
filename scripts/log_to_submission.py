import argparse
import glob
import json
import os
import shutil
from typing import Any, Dict, List, Optional

import datasets
import numpy as np


BENCHMARK_STORAGE: Optional[str] = "ai-forever/MERA"
_TASKS = {}


def get_files_from_dir(dir_path):
    f = []
    for dir_path, dirn_ames, filenames in os.walk(dir_path):
        for fn in filenames:
            fn = os.path.join(dir_path, fn)
            f.append(fn)
    return f


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(obj, file, ensure_ascii=False, indent=4)


def load_json(path):
    with open(path, encoding="utf-8") as file:
        text = json.loads(file.read().strip())
    return text


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
    def outputs_path(self):
        output_answers: List[str] = glob.glob(
            os.path.join(self.outputs_dir, f"*{self.src_name}.jsonl")
        )
        return output_answers[0] if len(output_answers) > 0 else ""

    @property
    def submission_path(self):
        return os.path.join(self.dst_dir, f"{self.dst_name}.json")

    @staticmethod
    def doc_to_meta(doc):
        return doc["meta"]

    def doc_to_id(self, doc):
        return self.doc_to_meta(doc)["id"]

    def load(self):
        if self.dataset_path is None or len(self.dataset_path) == 0:
            dataset = datasets.load_dataset(path=BENCHMARK_STORAGE, name=self.src_name)[
                "test"
            ]
        else:
            dataset = load_json(self.dataset_path)["data"]["test"]
        examples = dict()
        for example in dataset:
            doct_id = self.doc_to_id(example)
            examples[doct_id] = example
        return examples

    def __init__(self, outputs_dir, dst_dir, dataset_path: Optional[str] = None):
        self.outputs_dir = outputs_dir
        self.dst_dir = dst_dir
        self.dataset_path = dataset_path
        self.dataset = self.load()


class ClassificationTask(BaseTask):
    @property
    def choices(self):
        return ["0", "1"]

    def convert(self):
        submission = self.outputs_to_submission(load_json(self.outputs_path))
        save_json(submission, self.submission_path)
        return submission

    def outputs_to_submission(self, outputs):
        res = []
        for storage in outputs:
            doc_id = int(storage["doc"]["meta"]["id"])
            answers = [
                [resp_idx, resp]
                for resp_idx, resp in enumerate(storage["filtered_resps"])
            ]
            res.extend([self.doc_outputs_to_submission(doc_id, answers)])
        return {"data": {"test": res}}

    @staticmethod
    def parse_doc(doc):
        return doc[0], doc[1][0]

    def doc_outputs_to_submission(self, doc_id, outputs):
        log_probs = np.zeros(len(outputs))
        for doc in outputs:
            idx, prob = self.parse_doc(doc)
            log_probs[idx] = prob
        idx = log_probs.argmax()
        res = {
            "outputs": self.choices[idx],
            "meta": {"id": doc_id},
        }
        return res


class TextTask(ClassificationTask):
    def doc_outputs_to_submission(self, doc_id, outputs):
        res = {
            "outputs": outputs[0][1].strip(),
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
        text = outputs[0][1].strip()
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
            "outputs": outputs[0][1][-1].strip(),
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
            "outputs": outputs[0][1].strip(),
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
        if self.dataset_path is None or len(self.dataset_path) == 0:
            dataset = datasets.load_dataset(path=BENCHMARK_STORAGE, name=self.src_name)[
                "test"
            ]
            dataset = [list(dataset)]
        else:
            dataset = load_json(self.dataset_path)["data"]["test"]
        return dataset

    @property
    def choices(self):
        return ["1", "2"]

    def outputs_to_submission(
        self, outputs: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, list]]:
        res_by_qid: Dict[int, Dict[int, Any]] = dict()
        for storage in outputs:
            dialog_id = int(storage["doc"]["meta"]["dialog_id"])
            question_id = int(storage["doc"]["meta"]["question_id"])
            answers = [
                # resp depends on num dims of filtered_resps[0]
                # resp is supposed to be the same as for other loglike tasks
                [resp_idx, resp]
                for resp_idx, resp in enumerate(storage["filtered_resps"])
            ]
            res_by_qid.setdefault(dialog_id, dict())[
                question_id
            ] = self.doc_outputs_to_submission(question_id, answers)
        del dialog_id, question_id
        res = []
        not_provided_ids = []
        for dialog in self.dataset:
            new_dialog = []
            for question in dialog:
                dialog_id = question["meta"]["dialog_id"]
                question_id = question["meta"]["question_id"]
                question_id_outputs = res_by_qid.get(dialog_id, dict()).get(question_id)
                if question_id_outputs is not None:
                    new_question = {
                        "outputs": question_id_outputs,
                        "meta": {
                            "dialog_id": dialog_id,
                            "question_id": question_id,
                        },
                    }
                    new_dialog.extend([new_question])
                else:
                    not_provided_ids.extend([question_id])
            res.extend([new_dialog])
            if len(not_provided_ids) > 0:
                print("Question ids not provided:", not_provided_ids)

        return {"data": {"test": res}}

    def doc_outputs_to_submission(self, doc_id, outputs):
        log_probs = np.zeros(len(outputs))
        for doc in outputs:
            idx, prob = self.parse_doc(doc)
            log_probs[idx] = prob
        idx = log_probs.argmax()
        return self.choices[idx]


@register_task
class ruHumanEval(TextTask):
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
    parser.add_argument("--outputs_dir", type=str, help="lm harness outputs")
    parser.add_argument(
        "--dst_dir",
        type=str,
        default="submission/",
        help="dir to save files for submission",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        help="dir with datasets",
    )
    parser.add_argument(
        "--logs_public_submit",
        type=bool,
        default=True,
        help="pack logs for public submission in separate file",
        action=argparse.BooleanOptionalAction,  # type: ignore[attr-defined]
    )
    res = parser.parse_known_args()[0]
    return res


def _get_dataset_path_by_taskname(
    source_paths: List[str], task_name: str
) -> Optional[str]:
    dst = None
    for task_path in source_paths:
        # try resolve
        if task_name.lower() in task_path:
            dst = task_path
            print("Process", task_name, "dataset path", dst)
            break
        dst_task_name = os.path.split(os.path.split(task_path)[0])[-1].lower()
        k = len(set(task_name.lower()).intersection(set(dst_task_name))) / max(
            len(dst_task_name), len(task_name)
        )
        if 0.65 < k:
            dst = task_path
            print(
                "Process",
                task_name,
                "dataset path resolved from",
                dst,
                dst_task_name,
                k,
            )
            break
    return dst


def create_submission(outputs_dir, dst_dir, dataset_dir: Optional[str] = None):
    os.makedirs(dst_dir, exist_ok=True)
    dataset_dir_defined = False
    if dataset_dir is not None and len(dataset_dir) > 0:
        paths = [x for x in get_files_from_dir(dataset_dir) if x.endswith("task.json")]
        dataset_dir_defined = True
    no_tasks = []
    for task_name, task_cls in _TASKS.items():
        dst = (
            _get_dataset_path_by_taskname(paths, task_name)
            if dataset_dir_defined
            else None
        )
        if dst is None and dataset_dir_defined:
            print("Can't find", task_name)
            no_tasks.append(task_name)
        else:
            task = task_cls(outputs_dir=outputs_dir, dst_dir=dst_dir, dataset_path=dst)
            _ = task.convert()
        print("---------------------")
    print("Not refactored tasks", no_tasks)
    zip_path = shutil.make_archive(dst_dir, "zip", dst_dir)
    print("Submission stored at", zip_path)
    return no_tasks


def pack_submission_logs(outputs_dir: str, dst_dir: str):
    if os.path.isdir(outputs_dir):
        zip_dir = f"{dst_dir}_logs_public"
        os.makedirs(zip_dir, exist_ok=True)
        for file_path in glob.glob(os.path.join(outputs_dir, "*.json*")) + glob.glob(
            os.path.join(outputs_dir, "rutie/*.json*")
        ):
            shutil.copy2(file_path, zip_dir)
        zip_path = shutil.make_archive(zip_dir, "zip", zip_dir)
        shutil.rmtree(zip_dir)
        print("Logs to add with public submission stored at", zip_path)
    else:
        raise ValueError(f"{outputs_dir} is not directory")


def main():
    args = get_args()
    _ = create_submission(args.outputs_dir, args.dst_dir, args.dataset_dir)
    if args.logs_public_submit:
        print("Packing logs for public submission...")
        pack_submission_logs(args.outputs_dir, args.dst_dir)


if __name__ == "__main__":
    main()
