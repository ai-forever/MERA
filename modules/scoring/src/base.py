from src.utils import ensure_directory_exists, load_yaml
from src.logger_config import get_logger


class Base(object):
    @classmethod
    def name(cls):
        """Return name of class in lower case"""
        return cls.__name__.lower()

    @property
    def working_dir(self):
        return self.conf.args.working_dir

    def __init__(self, conf, log=None):
        super().__init__()
        self.conf = load_yaml(conf)
        self.name = self.name()
        if log is None:
            log = get_logger(self.conf.args.log_dir, self.name).info
        self.log_fn = log
        ensure_directory_exists(self.working_dir)

    def log(self, text):
        if self.conf.args.verbose:
            self.log_fn(text)
