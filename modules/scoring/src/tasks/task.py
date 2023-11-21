from src.base import Base
from src.utils import load_yaml, update_seed, load_json
from src.enums import Errors
from src.dataset import Dataset
from src.metrics import mean
from abc import ABCMeta, abstractmethod
from typing import Dict, Union
from collections import defaultdict
import numpy as np
import os
import traceback
import datasets


class Task(Base, metaclass=ABCMeta):

    @property
    def local_path(self):
        return self.task_conf.origin_repo_path

    @property
    def task_conf_path(self):
        return f"configs/{self.name}.yaml"

    @property
    def choices(self):
        return ["0", "1"]

    def __init__(self, conf):
        super().__init__(conf)
        self.task_conf = load_yaml(self.task_conf_path)
        self.gold: Union[Dataset, None] = None
        self.split = self.task_conf.split
        update_seed(self.conf.args.seed)
        self._aggregation = None

    def load_gold(self):
        ds = datasets.load_dataset(path="ai-forever/MERA", name=self.name.lower())["test"]
        examples = dict()
        for example in ds:
            doct_id = self.doc_to_id(example)
            examples[doct_id] = example
        self.gold = Dataset(
            local_path="",
            name=self.name,
            log=self.log,
            examples=examples
        )
        return []

    def load_and_validate(self, local_path):
        self.log(f"Load dataset {self.name}")
        dataset = None
        errors = []
        extension = os.path.splitext(local_path)[-1]
        if self.task_conf.extension != extension:
            self.log(f"{Errors.extension} {self.name}")
            errors.append({"type": str(Errors.extension), "extension": extension})
            return dataset, errors
        try:
            dataset = load_json(local_path)
        except:
            self.log(f"{Errors.unreadable_file} {self.name}")
            errors.append({
                "type": str(Errors.unreadable_file), "local_path": local_path, "trace": traceback.format_exc()})
            return dataset, errors
        if "data" not in dataset:
            self.log(f"{Errors.no_data_field} {self.name}")
            errors.append({"type": str(Errors.no_data_field)})
        elif self.split not in dataset["data"]:
            self.log(f"{Errors.no_split} {self.name}")
            errors.append({"type": str(Errors.no_split), "split": self.split})
        else:
            examples = dict()
            for idx, example in enumerate(dataset["data"][self.split]):
                try:
                    _ = self.doc_to_y_pred(example)
                except KeyError:
                    self.log(f"{Errors.no_outputs_field_for_doc} {self.name}")
                    errors.append({"type": str(Errors.no_outputs_field_for_doc), "example_number": idx})
                try:
                    _ = self.doc_to_meta(example)
                except KeyError:
                    self.log(f"{Errors.no_meta_field_for_doc} {self.name}")
                    errors.append({"type": str(Errors.no_meta_field_for_doc), "example_number": idx})
                    continue
                try:
                    doct_id = self.doc_to_id(example)
                except KeyError:
                    self.log(f"{Errors.no_id_field_for_doc} {self.name}")
                    errors.append({"type": str(Errors.no_id_field_for_doc), "example_number": idx})
                    continue
                examples[doct_id] = example
            if not len(errors):
                dataset = Dataset(
                    local_path=local_path,
                    name=self.name,
                    log=self.log,
                    examples=examples
                )
        return dataset, errors

    def evaluate(self, local_path):
        self.log(f"Start evaluating dataset {self.name}")
        dataset, errors = self.load_and_validate(local_path=local_path)
        vals = defaultdict(list)
        results = {}
        # Calculate metrics
        if not len(errors):
            for doc_id in self.gold.doc_ids():
                if doc_id not in dataset.examples:
                    self.log(f"{Errors.no_id} {self.name}")
                    errors.append({"type": str(Errors.no_id), "doc_id": doc_id})
                elif not isinstance(self.doc_to_y_pred(dataset[doc_id]), type(self.doc_to_y_true(self.gold[doc_id]))):
                    errors.append({"type": str(Errors.doc_output_type_error), "doc_id": doc_id})
                else:
                    try:
                        metrics = self.process_results(doc_true=self.gold[doc_id], doc_pred=dataset[doc_id])
                        for metric, value in metrics.items():
                            vals[metric].append(value)
                    except:
                        errors.append({
                            "type": str(Errors.doc_parse_output_error),
                            "doc_id": doc_id,
                            "trace": traceback.format_exc()
                        })
        # Aggregate metrics
        if not len(errors):
            for metric, items in vals.items():
                results[metric] = self.aggregation()[metric](items)
        return results, errors

    def doc_to_y_true(self, doc):
        return doc["outputs"]

    def doc_to_y_pred(self, doc):
        return doc["outputs"]

    def doc_to_meta(self, doc):
        return doc["meta"]

    def doc_to_id(self, doc):
        return self.doc_to_meta(doc)["id"]

    @abstractmethod
    def aggregation(self) -> Dict:
        raise NotImplemented

    @abstractmethod
    def process_results(self, doc_true, doc_pred) -> Dict:
        raise NotImplemented

    def sample_submission(self):
        res = []
        for doc_id in self.gold.doc_ids():
            doc = {
                "outputs": str(np.random.choice(self.choices)),
                "meta": {"id": doc_id}
            }
            res.append(doc)
        return {"data": {self.split: res}}

    def average_results(self, metrics: Dict) -> float:
        return mean([v for k, v in metrics.items() if "." not in k])


class NotImplementedTask(Task):

    def load_gold(self):
        return []

    @abstractmethod
    def evaluate(self, local_path):
        raise NotImplemented

    def sample_submission(self):
        return {"data": {"test": [{"outputs": "1", "inputs": "1", "meta": {"id": 1}}]}}
