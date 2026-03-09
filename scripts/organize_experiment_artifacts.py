#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
from pathlib import Path


LOG_DIR_RE = re.compile(r"Log directory:\s*(?:\./)?experiments/([^/\s]+)/([^/\s]+)/")
CHECKPOINT_RE = re.compile(r"checkpoint=(/.*/experiments/[^/\s]+/[^/\s]+)/[^/\s]+")
KNOWN_MODELS = {"e2cstp", "steve", "timecma", "timellm", "stllm"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/root/LargeST")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--active-stem", action="append", default=[])
    parser.add_argument("--reconcile-legacy-pids", action="store_true")
    return parser.parse_args()


def read_text(path):
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def detect_log_dir_from_text(root, text):
    match = LOG_DIR_RE.search(text)
    if match:
        return Path(root) / "experiments" / match.group(1) / match.group(2)
    match = CHECKPOINT_RE.search(text)
    if match:
        return Path(match.group(1))
    return None


def companion_candidates(path):
    stem = path.stem
    candidates = [stem]
    if stem.endswith(".nohup"):
        candidates.append(stem[: -len(".nohup")])
    if stem.endswith("_eval"):
        candidates.append(stem[: -len("_eval")])
    return list(dict.fromkeys(candidates))


def detect_run_dir_from_moved_log(path):
    if path.parent.name in {"launch", "eval", "embed", "metrics", "pid", "waiter"} and path.parent.parent.name == "artifacts":
        return path.parent.parent.parent
    return detect_log_dir_from_text(path.anchor or "/", read_text(path))


def find_moved_companion_log(root, stems):
    experiments_dir = Path(root) / "experiments"
    for stem in stems:
        matches = sorted(experiments_dir.rglob(stem + ".log"))
        if len(matches) == 1:
            found = matches[0]
            run_dir = detect_run_dir_from_moved_log(found)
            if run_dir is not None:
                return run_dir
    return None


def detect_target_dir(root, path):
    if path.suffix == ".json":
        try:
            payload = json.loads(read_text(path))
            checkpoint = payload.get("checkpoint")
            if checkpoint:
                checkpoint_path = Path(str(checkpoint))
                return checkpoint_path.parent
        except Exception:
            pass

    text = read_text(path)
    target_dir = detect_log_dir_from_text(root, text)
    if target_dir is not None:
        return target_dir

    for stem in companion_candidates(path):
        companion_log = path.with_name(stem + ".log")
        if companion_log.exists():
            target_dir = detect_log_dir_from_text(root, read_text(companion_log))
            if target_dir is not None:
                return target_dir
    moved_target = find_moved_companion_log(root, companion_candidates(path))
    if moved_target is not None:
        return moved_target

    prefix = path.name.split("_", 1)[0].lower()
    if prefix in KNOWN_MODELS:
        return Path(root) / "experiments" / prefix / "_legacy" / path.stem
    return Path(root) / "experiments" / "_misc" / "_legacy" / path.stem


def detect_bucket(path):
    stem = path.stem.lower()
    if path.suffix == ".pid":
        return "pid"
    if path.suffix == ".json":
        return "metrics"
    if "embed" in stem:
        return "embed"
    if "eval" in stem or "test" in stem:
        return "eval"
    if "waiter" in stem:
        return "waiter"
    return "launch"


def should_skip(path, active_stems):
    stem = path.stem
    for active in active_stems:
        if stem == active or stem.startswith(active + ".") or stem.startswith(active + "_"):
            return True
    return False


def move_file(path, dest, execute):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if execute:
        shutil.move(str(path), str(dest))


def main():
    args = parse_args()
    root = Path(args.root)
    active_stems = set(args.active_stem)
    results = []

    candidates = []
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix in {".log", ".pid", ".json"}:
            candidates.append(path)
    if args.reconcile_legacy_pids:
        candidates.extend(sorted((root / "experiments").rglob("artifacts/pid/*.pid")))

    for path in candidates:
        if should_skip(path, active_stems):
            results.append({"action": "skip-active", "source": str(path)})
            continue

        target_dir = detect_target_dir(root, path)
        bucket = detect_bucket(path)
        dest = target_dir / "artifacts" / bucket / path.name
        if dest.resolve() == path.resolve():
            results.append({"action": "skip-same", "source": str(path)})
            continue
        if dest.exists():
            results.append({"action": "skip-exists", "source": str(path), "dest": str(dest)})
            continue
        move_file(path, dest, args.execute)
        results.append({"action": "move", "source": str(path), "dest": str(dest)})

    print(json.dumps(results, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
