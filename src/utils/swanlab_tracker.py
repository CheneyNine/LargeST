import warnings

import numpy as np


class SwanLabTracker:
    def __init__(
        self,
        enabled,
        logger,
        project,
        experiment_name,
        config=None,
        mode="cloud",
        logdir="",
    ):
        self._enabled = bool(enabled)
        self._logger = logger
        self._module = None
        self._active = False
        self._warned_log_error = False

        if not self._enabled:
            return

        try:
            import swanlab
        except Exception as error:
            self._logger.warning(
                "SwanLab disabled because import failed: {}".format(error)
            )
            return

        init_kwargs = {
            "project": project,
            "experiment_name": experiment_name,
            "config": config or {},
        }
        if mode:
            init_kwargs["mode"] = mode
        if logdir:
            init_kwargs["logdir"] = logdir

        try:
            swanlab.init(**init_kwargs)
        except TypeError:
            # Fallback for older/newer SwanLab API variants.
            fallback_kwargs = {
                "project": project,
                "experiment_name": experiment_name,
                "config": config or {},
            }
            try:
                swanlab.init(**fallback_kwargs)
            except Exception as error:
                self._logger.warning(
                    "SwanLab disabled because init failed: {}".format(error)
                )
                return
        except Exception as error:
            self._logger.warning("SwanLab disabled because init failed: {}".format(error))
            return

        self._module = swanlab
        self._active = True
        self._logger.info(
            "SwanLab enabled. project={}, experiment={}".format(
                project, experiment_name
            )
        )

    def _normalize_value(self, value):
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, (np.floating, np.integer)):
            return float(value)
        if hasattr(value, "item"):
            try:
                return float(value.item())
            except Exception:
                return None
        return None

    def log(self, payload, step=None):
        if not self._active:
            return

        clean_payload = {}
        for key, value in payload.items():
            numeric = self._normalize_value(value)
            if numeric is not None:
                clean_payload[str(key)] = numeric

        if not clean_payload:
            return

        try:
            if step is None:
                self._module.log(clean_payload)
            else:
                self._module.log(clean_payload, step=int(step))
        except TypeError:
            # Some versions may not accept `step`.
            try:
                self._module.log(clean_payload)
            except Exception as error:
                if not self._warned_log_error:
                    self._warned_log_error = True
                    self._logger.warning(
                        "SwanLab logging failed (further errors muted): {}".format(error)
                    )
        except Exception as error:
            if not self._warned_log_error:
                self._warned_log_error = True
                self._logger.warning(
                    "SwanLab logging failed (further errors muted): {}".format(error)
                )

    def finish(self):
        if not self._active:
            return
        try:
            self._module.finish()
        except Exception as error:
            warnings.warn("SwanLab finish failed: {}".format(error))
        self._active = False
