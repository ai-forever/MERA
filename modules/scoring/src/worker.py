from src.base import Base
from src.tasks import get_all_tasks
from src.enums import Errors, SubmissionStatus
from src.utils import get_files_from_dir, save_json, update_seed, load_yaml
from src.metrics import mean
from collections import defaultdict
from typing import Dict
from copy import deepcopy
import random
import traceback
import zipfile
import os
import shutil


class Worker(Base):
    def __init__(self, conf, no_load_models=False):
        super().__init__(conf)
        self.tasks = None
        update_seed(self.conf.args.seed)
        self.comments = load_yaml(self.conf.args.errors_comments)
        self.conf.no_load_models = no_load_models

    def load(self):
        errors = defaultdict(list)
        tasks = {}
        for task_name, task_cls in get_all_tasks().items():
            self.log(f"Create task {task_name}")
            try:
                task = task_cls(conf=self.conf)
                task_errors = task.load_gold()
                if len(task_errors):
                    errors[task_name] = task_errors
                else:
                    tasks[task_name] = task
            except:
                self.log(f"System error while loading task {task_name}")
                errors[task_name].append({"type": str(Errors.task_system_error), "trace": traceback.format_exc()})
        self.tasks = tasks
        return errors

    @staticmethod
    def prepare(local_path: str) -> (Dict, Dict):
        """Return dict of {task_name: local_path}"""
        errors = defaultdict(list)
        files = {}
        try:
            if local_path.endswith(".zip"):
                dst_dir = local_path[:-4]
                os.makedirs(dst_dir, exist_ok=True)
                with zipfile.ZipFile(local_path, 'r') as zip_ref:
                    zip_ref.extractall(dst_dir)
                for x in get_files_from_dir(dst_dir):
                    task_name = os.path.splitext(os.path.split(x)[-1])[0]
                    x = os.path.abspath(x)
                    files[task_name] = x
            else:
                task_name = os.path.splitext(os.path.split(local_path)[-1])[0]
                files[task_name] = os.path.abspath(local_path)
        except:
            task_name = os.path.splitext(os.path.split(local_path)[-1])[0]
            error_key = "_all" if local_path.endswith(".zip") else task_name
            errors[error_key].append({"type": str(Errors.unreadable_zip), "trace": traceback.format_exc()})
        return files, errors

    def total_score(self, metrics):
        res = []
        for task_name in metrics:
            task = self.tasks[task_name]
            if task.task_conf.use_in_total:
                res.append(task.average_results(metrics[task_name]))
        return mean(res)

    def evaluate(self, local_path: str = None):
        self.log(f"Start evaluate {local_path}")
        errors = defaultdict(list)
        files, new_errors = self.prepare(local_path=local_path)
        errors = self._update_dict(errors, new_errors)
        results = {}
        if len(files) == 1:
            for task_name in files:
                task = self.tasks[task_name]
                task_res, task_errors = task.evaluate(files[task_name])
                if len(task_errors):
                    errors[task_name].extend(task_errors)
                else:
                    results[task_name] = task_res
        else:

            for task_name, task in self.tasks.items():
                if task_name not in files:
                    errors[task_name].append({"type": str(Errors.no_task)})
                else:
                    task_res, task_errors = task.evaluate(files[task_name])
                    if len(task_errors):
                        errors[task_name].extend(task_errors)
                    else:
                        results[task_name] = task_res
        if len(errors):
            _errors_reason, errors_for_user, global_error_reason = self.postprocess_errors(errors)
            _global_errors_reason = _errors_reason.pop("_all", {})
            res = {
                "status": str(SubmissionStatus.failed),
                "error_reason": dict(errors_for_user),
                "global_error_reason": dict(global_error_reason),
                "_errors_reason": dict(_errors_reason),
                "_global_errors_reason": dict(_global_errors_reason),
                "errors": dict(errors)
            }
        else:
            res = {"status": str(SubmissionStatus.ok)}
        results["total_score"] = self.total_score(results)
        res["results"] = results
        return res

    def postprocess_errors(self, errors):
        errors_for_user = {}
        for task_name in errors:
            task_errors = deepcopy(errors[task_name])
            for doc_error in task_errors:
                _ = doc_error.pop("trace", None)
                _ = doc_error.pop("s3_path", None)
                _ = doc_error.pop("local_path", None)
                doc_error["comment"] = self.comments[doc_error["type"]]
            errors_for_user[task_name] = task_errors
        global_error_reason = errors_for_user.pop("_all", {})
        return errors, errors_for_user, global_error_reason

    @staticmethod
    def _update_dict(old, new):
        for k, v in new.items():
            old[k].extend(v)
        return old

    def generate_random_baseline(self, sample_submission_dir=None, make_zip=False, produce_errors=False):
        if sample_submission_dir is None:
            sample_submission_dir = os.path.join(self.working_dir, self.conf.args.sample_submission_dir_name)
        os.makedirs(sample_submission_dir, exist_ok=True)
        result = {}
        iters = list(self.tasks.items())
        if produce_errors:
            # Error.no_task
            iters.pop(random.randint(0, len(iters) - 1))
        rd = random.sample(range(len(iters)), 9)
        extension_error_task = rd[0]
        no_data_field = rd[1]
        no_split = rd[2]
        no_outputs_field_for_doc = rd[3]
        no_meta_field_for_doc = rd[8]
        no_id_field_for_doc = rd[5]
        doc_output_type_error = rd[6]
        no_id = rd[7]
        for idx, (task_name, task) in enumerate(iters):
            try:
                submission = task.sample_submission()
            except:
                self.log(f"Error while sample_submission for task {task_name}. Trace:\n{traceback.format_exc()}")
                continue
            extension = task.task_conf.extension
            if produce_errors:
                # Errors.no_data_field
                if idx == no_data_field:
                    submission = {}
                # Errors.no_split
                if idx == no_split:
                    submission = {"data": {task.task_conf.split + "_": submission["data"][task.task_conf.split]}}
                # Errors.extension
                if idx == extension_error_task:
                    extension = extension[:-1]
                # Errors.no_outputs_field_for_doc
                if idx == no_outputs_field_for_doc:
                    corrupt_idx = random.randint(0, len(submission["data"][task.task_conf.split]) - 1)
                    submission["data"][task.task_conf.split][corrupt_idx].pop("outputs")
                # Errors.no_outputs_field_for_doc
                if idx == no_meta_field_for_doc:
                    corrupt_idx = random.randint(0, len(submission["data"][task.task_conf.split]) - 1)
                    submission["data"][task.task_conf.split][corrupt_idx].pop("meta")
                # Errors.no_id_field_for_doc
                if idx == no_id_field_for_doc:
                    corrupt_idx = random.randint(0, len(submission["data"][task.task_conf.split]) - 1)
                    submission["data"][task.task_conf.split][corrupt_idx]["meta"].pop("id")
                # Errors.doc_output_type_error
                if idx == doc_output_type_error:
                    corrupt_idx = random.randint(0, len(submission["data"][task.task_conf.split]) - 1)
                    submission["data"][task.task_conf.split][corrupt_idx]["outputs"] = -777
                # Errors.no_id
                if idx == no_id:
                    corrupt_idx = random.randint(0, len(submission["data"][task.task_conf.split]) - 1)
                    submission["data"][task.task_conf.split].pop(corrupt_idx)

            submission_path = os.path.join(sample_submission_dir, f"{task_name}{extension}")
            try:
                save_json(submission, submission_path)
            except:
                self.log(f"Error while save  submission for task {task_name}")
            result[task_name] = submission_path
        zip_submission_path = None
        if make_zip:
            zip_submission_path = shutil.make_archive(sample_submission_dir, "zip", sample_submission_dir)
        return result, zip_submission_path
