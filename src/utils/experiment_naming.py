import os
import re


_TOKEN_RE = re.compile(r"[^A-Za-z0-9._+-]+")


def sanitize_experiment_token(value):
    text = str(value).strip()
    if not text:
        return "na"
    text = _TOKEN_RE.sub("_", text)
    text = text.strip("._-")
    return text or "na"


def build_experiment_dir_name(dataset, years, seq_len, horizon, seed, extra_parts=None, run_tag=""):
    parts = [
        "ds-{}".format(sanitize_experiment_token(dataset)),
        "yr-{}".format(sanitize_experiment_token(years)),
        "q{}".format(int(seq_len)),
        "h{}".format(int(horizon)),
        "s{}".format(int(seed)),
    ]
    for label, value in extra_parts or []:
        if value is None:
            continue
        parts.append(
            "{}-{}".format(
                sanitize_experiment_token(label),
                sanitize_experiment_token(value),
            )
        )
    if str(run_tag).strip():
        parts.append("tag-{}".format(sanitize_experiment_token(run_tag)))
    return "__".join(parts)


def get_artifact_dir(log_dir, kind):
    return os.path.join(log_dir, "artifacts", sanitize_experiment_token(kind))
