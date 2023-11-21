import importlib
import os


_TASKS = dict()


def register_all(services_dir):
    # import any Python files in the services_dir directory
    # DO NOT MOVE THIS FUNCTION TO OTHER DIRECTORY
    branches = []
    for branch in os.walk(services_dir):
        branches.append(branch)
    files = []
    for address, _, dir_files in branches:
        for file in dir_files:
            files.append(os.path.join(address, file))
    for path in files:
        path, file = os.path.split(path)
        if (
            not file.startswith('_')
            and not file.startswith('.')
            and file.endswith('.py')
        ):
            module_name = file[:-3]
            module = f"src.tasks.{module_name}"
            # print(f"import module {module}")
            _ = importlib.import_module(f"{module}")


def register_task(cls):
    name = cls.__name__
    _TASKS[name] = cls
    return cls


def rel_dir(file):
    return os.path.split(file)[0]


def get_all_tasks():
    return _TASKS
