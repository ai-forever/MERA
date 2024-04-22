from typing import Callable, Optional

from lm_eval.api.instance import Instance


class ContextInstance(Instance):
    def __init__(
        self,
        requests_updater: Optional[Callable] = None,
        storage_updater: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._update_request = requests_updater
        self._update_storage = storage_updater

    @property
    def update_request(self):
        if getattr(self, "_update_request") is not None:
            return self._update_request
        raise NotImplementedError("Method for updating request is not defined.")

    @property
    def update_storage(self):
        if getattr(self, "_update_storage") is not None:
            return self._update_storage
        raise NotImplementedError("Method for updating storage is not defined.")
