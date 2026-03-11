import os
import re
from datetime import datetime


_TOKEN_RE = re.compile(r"[^A-Za-z0-9._+-]+")
_TIMESTAMP_FORMATS = ("%y%m%d%H%M", "%Y%m%d%H%M", "%y%m%d%H%M%S", "%Y%m%d%H%M%S")


def sanitize_experiment_token(value):
    text = str(value).strip()
    if not text:
        return "na"
    text = _TOKEN_RE.sub("_", text)
    text = text.strip("._-")
    return text or "na"


def _build_dataset_token(dataset, years):
    dataset_token = sanitize_experiment_token(dataset)
    years_token = sanitize_experiment_token(years)
    if years_token == "na":
        return dataset_token
    if dataset_token.lower().endswith(years_token.lower()):
        return dataset_token
    return "{}{}".format(dataset_token, years_token)


def build_run_timestamp(started_at=None):
    if started_at is None:
        started_at = datetime.now()
    elif isinstance(started_at, str):
        value = started_at.strip()
        parsed = None
        for fmt in _TIMESTAMP_FORMATS:
            try:
                parsed = datetime.strptime(value, fmt)
                break
            except ValueError:
                continue
        if parsed is None:
            raise ValueError("Invalid experiment timestamp: {}".format(started_at))
        started_at = parsed
    return started_at.strftime("%y%m%d%H%M")


def build_experiment_dir_name(
    model_name,
    dataset,
    years,
    seq_len,
    horizon,
    seed,
    extra_parts=None,
    run_tag="",
    started_at=None,
):
    del extra_parts
    del run_tag
    parts = [
        sanitize_experiment_token(model_name),
        _build_dataset_token(dataset, years),
        "q{}".format(int(seq_len)),
        "h{}".format(int(horizon)),
        "s{}".format(int(seed)),
        "t{}".format(build_run_timestamp(started_at)),
    ]
    return "_".join(parts)


def get_artifact_dir(log_dir, kind):
    return os.path.join(log_dir, "artifacts", sanitize_experiment_token(kind))
