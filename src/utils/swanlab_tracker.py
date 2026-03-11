import os
import warnings

import numpy as np


def resolve_swanlab_job_type(mode):
    mode = str(mode or "").strip().lower()
    if mode == "train":
        return "train"
    if mode in {"test", "val", "eval", "evaluate"}:
        return "eval"
    return "inference"


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
        description="",
        job_type="train",
        lark_webhook_url="",
        lark_secret="",
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

        callbacks = []
        resolved_webhook = str(
            lark_webhook_url or os.getenv("SWANLAB_LARK_WEBHOOK_URL", "")
        ).strip()
        resolved_secret = str(
            lark_secret or os.getenv("SWANLAB_LARK_SECRET", "")
        ).strip()
        if resolved_webhook:
            try:
                from swanlab.plugin.notification import LarkCallback

                callbacks.append(
                    LarkCallback(
                        webhook_url=resolved_webhook,
                        secret=resolved_secret,
                    )
                )
            except Exception as error:
                self._logger.warning(
                    "SwanLab Lark callback disabled because init failed: {}".format(error)
                )

        base_kwargs = {
            "project": project,
            "experiment_name": experiment_name,
            "config": config or {},
        }
        optional_kwargs = {}
        if mode:
            optional_kwargs["mode"] = mode
        if logdir:
            optional_kwargs["logdir"] = logdir
        if str(description).strip():
            optional_kwargs["description"] = str(description).strip()
        if str(job_type).strip():
            optional_kwargs["job_type"] = str(job_type).strip()
        if callbacks:
            optional_kwargs["callbacks"] = callbacks

        attempt_kwargs = [
            dict(base_kwargs, **optional_kwargs),
            dict(base_kwargs, **{k: v for k, v in optional_kwargs.items() if k != "callbacks"}),
            dict(base_kwargs, **{k: v for k, v in optional_kwargs.items() if k != "logdir"}),
            dict(
                base_kwargs,
                **{
                    k: v
                    for k, v in optional_kwargs.items()
                    if k not in {"mode", "logdir", "callbacks"}
                }
            ),
            dict(base_kwargs),
        ]

        last_error = None
        for init_kwargs in attempt_kwargs:
            try:
                swanlab.init(**init_kwargs)
                last_error = None
                break
            except Exception as error:
                last_error = error
                continue

        if last_error is not None:
            self._logger.warning("SwanLab disabled because init failed: {}".format(last_error))
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
