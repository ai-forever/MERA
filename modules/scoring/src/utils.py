from omegaconf import OmegaConf
import os
import json
import random
import numpy as np
import pickle


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def ensure_directory_exists(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)


def load_yaml(conf):
    if isinstance(conf, str):
        conf = OmegaConf.load(conf)
    return conf


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(obj, file, ensure_ascii=False, indent=4)


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        text = json.loads(file.read().strip())
    return text


def get_files_from_dir(dir_path):
    f = []
    for dir_path, dirn_ames, filenames in os.walk(dir_path):
        for fn in filenames:
            fn = os.path.join(dir_path, fn)
            f.append(fn)
    return f


def update_seed(seed=1234):
    import torch

    torch.manual_seed(seed)
    random.seed(10)
    np.random.seed(seed)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def random_choice(arr, size=1):
    ids = list(range(len(arr)))
    ids = np.random.choice(ids, size=size)
    if 1 == size:
        ids = [ids]
    res = [arr[int(idx)] for idx in ids]
    return res
