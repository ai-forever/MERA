class Dataset(object):
    def __init__(self, local_path: str, name: str, log, examples: dict):
        self.local_path = local_path
        self.name = name
        self.log = log
        self.examples = examples

    def doc_ids(self):
        return list(self.examples.keys())

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]
