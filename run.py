"""Run one independent CPU trial from the compact source release.

Examples:
    python run.py --problem CVRP --instance A-n32-k5 --iterations 100
    python run.py --problem VRPTW --instance C101 --seconds 10

Neural methods use random weights by default. An optional compatible checkpoint
changes initialization. These new runs are separate from the archived results.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import re
import sys
import tempfile


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "reproduced"
METHODS = ("ALNS", "AC", "DAC-Partial", "DAC-Pre", "DAC-Credit")
sys.dont_write_bytecode = True


def positive_integer(value):
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def positive_seconds(value):
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a finite positive number") from exc
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def instance_entry(problem, identifier):
    manifest = json.loads((ROOT / "configs/instances.json").read_text(encoding="utf-8-sig"))
    if isinstance(manifest, dict):
        manifest = manifest.get("instances", manifest.get("entries"))
    if not isinstance(manifest, list):
        raise ValueError("configs/instances.json must contain an instance list")
    matches = [row for row in manifest if str(row.get("problem", "")).upper() == problem
               and str(row.get("id", "")).casefold() == identifier.casefold()]
    if len(matches) != 1:
        raise ValueError(f"Expected one {problem} instance named {identifier}; found {len(matches)}")
    row = matches[0]
    relative = str(row["path"]).replace("\\", "/")
    if relative.startswith("/") or PureWindowsPath(relative).is_absolute():
        raise ValueError("Instance manifest paths must be relative to this release")
    path = (ROOT / relative).resolve()
    if not path.is_relative_to((ROOT / "instances").resolve()):
        raise ValueError("Instance path must stay within instances/")
    if not path.is_file():
        raise FileNotFoundError("Missing instance: " + relative)
    expected = str(row.get("sha256", "")).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("Instance manifest has no valid SHA256")
    if sha256(path) != expected:
        raise ValueError("Instance SHA256 differs from configs/instances.json")
    return row, path


def redirect_runtime_caches():
    """Keep generated files beside the run outputs and outside the source tree."""
    cache = OUTPUT / ".cache"
    locations = {
        "MPLCONFIGDIR": cache / "matplotlib",
        "NUMBA_CACHE_DIR": cache / "numba",
        "TORCH_HOME": cache / "torch",
        "TORCH_EXTENSIONS_DIR": cache / "torch_extensions",
        "TORCHINDUCTOR_CACHE_DIR": cache / "torch_inductor",
        "XDG_CACHE_HOME": cache,
        "TMP": cache / "tmp",
        "TEMP": cache / "tmp",
        "TMPDIR": cache / "tmp",
    }
    for name, path in locations.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    tempfile.tempdir = str(cache / "tmp")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--problem", type=str.upper, choices=("CVRP", "VRPTW"), required=True)
    parser.add_argument("--instance", required=True, help="Instance ID, case insensitive")
    parser.add_argument("--method", choices=METHODS, default="DAC-Partial")
    parser.add_argument("--seed", type=int, default=0)
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--iterations", type=positive_integer, help="Positive iteration count; default 100")
    budget.add_argument("--seconds", type=positive_seconds, help="Positive wall-clock budget in seconds")
    parser.add_argument("--checkpoint", type=Path, help="Optional compatible AC/DAC checkpoint; default random weights")
    args = parser.parse_args()
    if not 0 <= args.seed < 2 ** 32:
        parser.error("--seed must be between 0 and 4294967295")
    if args.checkpoint and args.method == "ALNS":
        parser.error("--checkpoint applies to neural methods only")
    now = datetime.now(timezone.utc)
    record = {
        "record_type": "independent_run",
        "independent_run": True,
        "formal_archive_record": False,
        "created_at_utc": now.isoformat(timespec="seconds"),
        "status": "error",
        "problem": args.problem,
        "method": args.method,
        "requested_instance": args.instance,
        "seed": args.seed,
        "initialization": "checkpoint" if args.checkpoint else "random_from_scratch" if args.method != "ALNS" else "heuristic_only",
        "notes": ["This independent run is separate from the archived formal observations.",
                  "Without a checkpoint, VRPTW neural initialization differs from the pretrained formal experiments.",
                  "CPU, one torch thread; raw Euclidean distance; fleet size is reported, not constrained."],
    }
    exit_code = 1
    try:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        redirect_runtime_caches()
        row, path = instance_entry(args.problem, args.instance)
        record["instance"] = row["id"]
        record["input"] = {"path": path.relative_to(ROOT).as_posix(), "sha256": row["sha256"],
                           "group": row.get("group"), "n_customers": row.get("n_customers")}
        config = {
            "problem": args.problem, "method": args.method, "instance_path": str(path),
            "instance_id": row["id"], "seed": args.seed, "device": "cpu",
            "budget_mode": "time" if args.seconds is not None else "iterations",
            "budget_value": args.seconds if args.seconds is not None else args.iterations or 100,
            "rollout_size": 8, "log_stride": 10, "record_parameter_hash": args.method != "ALNS",
        }
        if args.checkpoint:
            checkpoint = args.checkpoint.expanduser().resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError("Checkpoint does not exist: " + str(checkpoint))
            config["checkpoint_path"] = str(checkpoint)
            config["checkpoint_sha256"] = sha256(checkpoint)
        record["config"] = config
        sys.path.insert(0, str(ROOT / "src"))
        from controlled_engine import run_trial, warmup_backend
        from validate_routes import validate_routes

        warmup_backend(args.problem)
        result = run_trial(config)
        record["result"] = result
        validation = validate_routes(path, args.problem,
            {"routes": result.get("routes", []), "id_space": result.get("route_id_space", "internal")})
        objective = result.get("best_objective")
        matches = (objective is not None and math.isfinite(float(objective))
                   and abs(float(objective) - validation["distance"]) <= max(1e-3, abs(validation["distance"]) * 1e-6))
        validation["reported_objective_matches"] = matches
        if not matches:
            validation["errors"].append("Reported objective differs from independently computed route distance")
            validation["valid"] = False
        record["validation"] = validation
        record["status"] = "ok" if result.get("status") == "ok" and validation["valid"] else "validation_failed"
        exit_code = 0 if record["status"] == "ok" else 1
    except Exception as exc:
        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(record.get("instance", args.instance)))[:100]
    name = f"{now.strftime('%Y%m%dT%H%M%S%fZ')}_{args.problem}_{label}_{args.method}_s{args.seed}.json"
    destination = OUTPUT / name
    OUTPUT.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary = {"record_type": "independent_run", "status": record["status"], "output": destination.relative_to(ROOT).as_posix()}
    if "result" in record:
        summary.update(objective=record["result"].get("best_objective"), iterations=record["result"].get("iterations"))
    if "validation" in record:
        summary["routes_valid"] = record["validation"]["valid"]
    if "error" in record:
        summary["error"] = record["error"]
    print(json.dumps(summary, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
