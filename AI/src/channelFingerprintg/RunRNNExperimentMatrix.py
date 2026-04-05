#!/home/Research/.venvs/siwn-tf216/bin/python
import argparse
import csv
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_HOME_DIR = "/home/Research/POWDER/"
DEFAULT_PYTHON_BIN = "/home/Research/.venvs/siwn-tf216/bin/python"
DEFAULT_REFRESH_SECONDS = 30
DEFAULT_MIN_FREE_MEMORY_MIB = 16384
DEFAULT_MAX_UTILIZATION = 20
HISTORICAL_QUAD_BASELINE = "QuadrupletNet_Spectrogram_FeatureExtractor_RNN_specmagnitude_phase_in256_out128_alpha0.5_beta0.5_gamma0.2_RMSprop_lr0.0001_Sinusoid-Powder-OTA-Lab-Nodes+Sionna_13_1774908010.keras"
THIS_FILE = Path(__file__).resolve()

@dataclass
class ExperimentSpec:
    job_name: str
    family: str
    network_type: str
    signal_type: str = "Sinusoid"
    sample_source: str = "iq"
    augment_with_sionna: bool = True
    data_type: str = "Spectrogram"
    spectrogram_processing: str = "magnitude_phase"
    fft_len: int = 256
    output_length: int = 128
    optimizer: str = "RMSprop"
    learning_rate: float = 0.0001
    quantization_layer: bool = True
    seed: int = 42
    python_bin: str = DEFAULT_PYTHON_BIN
    batch_size: int = 128
    validation_size: float = 0.1
    epochs: int = 1000
    patience: int = 200
    factor: float = 0.1
    alpha: float | None = None
    beta: float | None = None
    gamma: float | None = None
    margin: float | None = None
    model_count: int = 13


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run parallel Sinusoid RNN training/evaluation jobs.")
    parser.add_argument("--home-dir", default=DEFAULT_HOME_DIR)
    parser.add_argument("--python-bin", default=DEFAULT_PYTHON_BIN)
    parser.add_argument("--gpu-ids", default="", help="Comma-separated physical GPU ids to use. If empty, auto-discover eligible GPUs.")
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--min-free-memory-mib", type=int, default=DEFAULT_MIN_FREE_MEMORY_MIB)
    parser.add_argument("--max-utilization", type=int, default=DEFAULT_MAX_UTILIZATION)
    parser.add_argument("--refresh-seconds", type=int, default=DEFAULT_REFRESH_SECONDS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker-job-json", default="", help=argparse.SUPPRESS)
    return parser.parse_args()


def ensure_expected_runtime(expected_python: str) -> None:
    actual = str(Path(sys.executable).resolve())
    expected = str(Path(expected_python).resolve())
    if actual != expected:
        raise SystemExit(
            f"This runner must be started with {expected}, but current interpreter is {actual}."
        )


def discover_gpus() -> List[Dict[str, Any]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    gpus = []
    for raw_line in result.stdout.strip().splitlines():
        if not raw_line.strip():
            continue
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) != 5:
            continue
        gpus.append({
            "index": int(parts[0]),
            "name": parts[1],
            "memory_total_mib": int(parts[2]),
            "memory_free_mib": int(parts[3]),
            "utilization_pct": int(parts[4]),
        })
    return gpus


def parse_gpu_ids(raw: str) -> List[int]:
    if not raw.strip():
        return []
    return [int(piece.strip()) for piece in raw.split(",") if piece.strip()]

def normalize_experiment_spec(experiment_spec: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in experiment_spec.items() if value is not None}

def validate_requested_gpu_ids(
    discovered: List[Dict[str, Any]],
    requested_ids: List[int],
) -> tuple[List[int], List[int]]:
    if not requested_ids:
        return [], []
    discovered_ids = {gpu["index"] for gpu in discovered}
    valid_ids: List[int] = []
    invalid_ids: List[int] = []
    seen_ids: set[int] = set()
    for gpu_id in requested_ids:
        if gpu_id in seen_ids:
            continue
        seen_ids.add(gpu_id)
        if gpu_id in discovered_ids:
            valid_ids.append(gpu_id)
        else:
            invalid_ids.append(gpu_id)
    return valid_ids, invalid_ids


def eligible_gpu_ids(
    discovered: List[Dict[str, Any]],
    requested_ids: List[int],
    busy_gpu_ids: set[int],
    min_free_memory_mib: int,
    max_utilization: int,
) -> List[int]:
    if requested_ids:
        return [gpu_id for gpu_id in requested_ids if gpu_id not in busy_gpu_ids]
    eligible = [
        gpu for gpu in discovered
        if gpu["index"] not in busy_gpu_ids
        and gpu["memory_free_mib"] >= min_free_memory_mib
        and gpu["utilization_pct"] <= max_utilization
    ]
    eligible.sort(key=lambda gpu: (gpu["memory_free_mib"], -gpu["utilization_pct"]), reverse=True)
    return [gpu["index"] for gpu in eligible]


def build_default_experiment_specs(python_bin: str) -> List[ExperimentSpec]:
    return [
        ExperimentSpec(
            job_name="hashnet_rnn_canonical",
            family="HashNet",
            network_type="RNN",
            margin=0.5,
            python_bin=python_bin,
        ),
        ExperimentSpec(
            job_name="tripletnet_rnn_canonical",
            family="TripletNet",
            network_type="RNN",
            alpha=0.5,
            beta=0.5,
            python_bin=python_bin,
        ),
        ExperimentSpec(
            job_name="quadrupletnet_rnn_canonical",
            family="QuadrupletNet",
            network_type="RNN",
            alpha=0.5,
            beta=0.5,
            gamma=0.0,
            python_bin=python_bin,
        ),
        ExperimentSpec(
            job_name="rnn_simple_baseline",
            family="RNN-Simple",
            network_type="RNN-Simple",
            alpha=0.5,
            beta=0.5,
            gamma=0.0,
            python_bin=python_bin,
        ),
    ]


def find_latest_artifact(models_dir: Path, family: str, prefer_sionna: bool = True) -> str | None:
    candidates = [path for path in models_dir.iterdir() if path.suffix == ".keras"]
    if family == "HashNet":
        candidates = [path for path in candidates if "HashNet" in path.name and "_FeatureExtractor_RNN_" in path.name]
    elif family == "TripletNet":
        candidates = [path for path in candidates if "TripletNet" in path.name and "_FeatureExtractor_RNN_" in path.name]
    elif family == "QuadrupletNet":
        candidates = [path for path in candidates if "QuadrupletNet" in path.name and "_FeatureExtractor_RNN_" in path.name]
    elif family == "RNN-Simple":
        candidates = [path for path in candidates if "_RNN-Simple_" in path.name]
    else:
        return None
    candidates = [
        path for path in candidates
        if "Sinusoid" in path.name
        and "_out128_" in path.name
        and "_specmagnitude_phase_" in path.name
    ]
    if prefer_sionna:
        preferred = [path for path in candidates if "+Sionna" in path.name]
        if preferred:
            candidates = preferred
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime)
    return candidates[-1].name


def build_jobs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    models_dir = Path(args.home_dir) / "Models"
    jobs: List[Dict[str, Any]] = []
    seen_artifacts = set()
    for family in ["HashNet", "TripletNet", "QuadrupletNet", "RNN-Simple"]:
        artifact = find_latest_artifact(models_dir, family)
        if artifact and artifact not in seen_artifacts:
            jobs.append({
                "job_id": f"baseline_{family.lower().replace('-', '_')}",
                "job_name": f"baseline_{family}",
                "job_type": "baseline_eval",
                "family": family,
                "artifact_name": artifact,
                "artifact_path": str(models_dir / artifact),
                "signal_type": "Sinusoid",
                "sample_source": "iq",
                "preprocessing": "magnitude_phase",
                "home_dir": args.home_dir,
                "python_bin": args.python_bin,
                "historical_baseline": False,
            })
            seen_artifacts.add(artifact)
    historical_path = models_dir / HISTORICAL_QUAD_BASELINE
    if historical_path.exists() and historical_path.name not in seen_artifacts:
        jobs.append({
            "job_id": "historical_quad_baseline",
            "job_name": "historical_quad_gamma_0_2",
            "job_type": "baseline_eval",
            "family": "QuadrupletNet",
            "artifact_name": historical_path.name,
            "artifact_path": str(historical_path),
            "signal_type": "Sinusoid",
            "sample_source": "iq",
            "preprocessing": "magnitude_phase",
            "home_dir": args.home_dir,
            "python_bin": args.python_bin,
            "historical_baseline": True,
        })
    for spec in build_default_experiment_specs(args.python_bin):
        jobs.append({
            "job_id": spec.job_name,
            "job_name": spec.job_name,
            "job_type": "train_eval",
            "family": spec.family,
            "experiment_spec": normalize_experiment_spec(asdict(spec)),
            "signal_type": spec.signal_type,
            "sample_source": spec.sample_source,
            "preprocessing": spec.spectrogram_processing,
            "home_dir": args.home_dir,
            "python_bin": args.python_bin,
            "historical_baseline": False,
        })
    return jobs


def build_worker_env(python_bin: str, gpu_id: int) -> Dict[str, str]:
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["VIRTUAL_ENV"] = str(Path(python_bin).resolve().parent.parent)
    env["PATH"] = str(Path(python_bin).resolve().parent) + os.pathsep + env.get("PATH", "")
    env["PYTHONNOUSERSITE"] = "1"
    return env


def build_failure_payload(
    job: Dict[str, Any],
    *,
    phase: str,
    error_message: str,
    traceback_text: str | None = None,
    artifact_name: str | None = None,
    artifact_path: str | None = None,
    training_result: Dict[str, Any] | None = None,
    return_code: int | None = None,
    log_path: Path | None = None,
) -> Dict[str, Any]:
    return {
        "job_id": job["job_id"],
        "job_name": job["job_name"],
        "job_type": job["job_type"],
        "family": job["family"],
        "artifact_name": artifact_name or job.get("artifact_name"),
        "artifact_path": artifact_path or job.get("artifact_path") or (training_result or {}).get("model_path"),
        "gpu_id": job.get("gpu_id"),
        "python_bin": sys.executable,
        "historical_baseline": bool(job.get("historical_baseline", False)),
        "training": training_result,
        "evaluation": None,
        "status": "failed",
        "error": {
            "phase": phase,
            "message": error_message,
            "traceback": traceback_text,
            "return_code": return_code,
            "log_path": str(log_path) if log_path else None,
        },
    }

def run_worker(job: Dict[str, Any]) -> Dict[str, Any]:
    print(f"[worker] python={sys.executable}")
    print(f"[worker] visible_gpu={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
    artifact_name = job.get("artifact_name")
    training_result = None
    try:
        if job["job_type"] == "train_eval":
            from TrainChannelFingerprinting import run_training_experiment
            training_result = run_training_experiment(
                job["experiment_spec"],
                home_dir=job["home_dir"],
                generate_dataset_plots=False,
            )
            artifact_name = training_result["model_name"]
        from TestChannelFingerprinting import build_standard_test_environments, evaluate_model_across_scenarios
        evaluation = evaluate_model_across_scenarios(
            artifact_name,
            scenario_environments=build_standard_test_environments(job["signal_type"]),
            home=job["home_dir"],
            signal_type=job["signal_type"],
            sample_source=job["sample_source"],
            preprocessing=job["preprocessing"],
        )
        payload = {
            "job_id": job["job_id"],
            "job_name": job["job_name"],
            "job_type": job["job_type"],
            "family": job["family"],
            "artifact_name": artifact_name,
            "artifact_path": job.get("artifact_path") or (training_result or {}).get("model_path"),
            "gpu_id": job.get("gpu_id"),
            "python_bin": sys.executable,
            "historical_baseline": bool(job.get("historical_baseline", False)),
            "training": training_result,
            "evaluation": evaluation,
            "status": "completed",
            "error": None,
        }
    except Exception as exc:
        payload = build_failure_payload(
            job,
            phase="worker",
            error_message=str(exc),
            traceback_text=traceback.format_exc(),
            artifact_name=artifact_name,
            training_result=training_result,
        )
        Path(job["result_path"]).write_text(json.dumps(payload, indent=2))
        raise
    result_path = Path(job["result_path"])
    result_path.write_text(json.dumps(payload, indent=2))
    return payload


def write_job_file(job: Dict[str, Any], runs_dir: Path) -> Path:
    job_path = runs_dir / f"{job['job_id']}.job.json"
    job_path.write_text(json.dumps(job, indent=2))
    return job_path


def launch_worker(job: Dict[str, Any], gpu_id: int, runs_dir: Path, python_bin: str) -> Dict[str, Any]:
    job = dict(job)
    job["gpu_id"] = gpu_id
    result_path = runs_dir / f"{job['job_id']}.result.json"
    if result_path.exists():
        result_path.unlink()
    job["result_path"] = str(result_path)
    job_path = write_job_file(job, runs_dir)
    log_path = runs_dir / f"{job['job_id']}.log"
    log_handle = log_path.open("w")
    proc = subprocess.Popen(
        [python_bin, str(THIS_FILE), "--worker-job-json", str(job_path)],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=build_worker_env(python_bin, gpu_id),
        cwd=str(THIS_FILE.parent),
        text=True,
    )
    return {
        "process": proc,
        "log_handle": log_handle,
        "log_path": log_path,
        "job": job,
        "job_path": job_path,
        "gpu_id": gpu_id,
    }


def load_result_payload(result_path: Path) -> Dict[str, Any]:
    return json.loads(result_path.read_text())


def build_leaderboard_rows(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for payload in results:
        if payload.get("status", "completed") != "completed" or not payload.get("evaluation"):
            continue
        evaluation = payload["evaluation"]
        aggregate = evaluation["aggregate"]
        row = {
            "job_id": payload["job_id"],
            "job_name": payload["job_name"],
            "job_type": payload["job_type"],
            "family": payload["family"],
            "artifact_name": payload["artifact_name"],
            "artifact_path": payload["artifact_path"],
            "gpu_id": payload.get("gpu_id"),
            "python_bin": payload["python_bin"],
            "historical_baseline": payload.get("historical_baseline", False),
            "reused_existing": (payload.get("training") or {}).get("reused_existing") if payload.get("training") else True,
            "admissible": aggregate["admissible"],
            "mean_KDR_AliceBob": aggregate["mean_KDR_AliceBob"],
            "mean_KDR_BobEve": aggregate["mean_KDR_BobEve"],
            "mean_KDR_AliceEve": aggregate["mean_KDR_AliceEve"],
            "mean_ones_fraction": aggregate["mean_ones_fraction"],
            "min_eve_gap": aggregate["min_eve_gap"],
        }
        for scenario in evaluation["per_scenario"]:
            sid = scenario["scenario_id"]
            row[f"{sid}_KDR_AliceBob"] = scenario["KDR_AliceBob"]
            row[f"{sid}_KDR_BobEve"] = scenario["KDR_BobEve"]
            row[f"{sid}_KDR_AliceEve"] = scenario["KDR_AliceEve"]
            row[f"{sid}_ones_fraction"] = scenario["ones_fraction"]
        rows.append(row)
    return rows


def choose_winner(rows: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    if not rows:
        return None
    admissible = [row for row in rows if row["admissible"]]
    if admissible:
        return sorted(admissible, key=lambda row: (row["mean_KDR_AliceBob"], -row["min_eve_gap"]))[0]
    return sorted(rows, key=lambda row: (-row["min_eve_gap"], row["mean_KDR_AliceBob"]))[0]


def write_leaderboard(rows: List[Dict[str, Any]], results_dir: Path) -> Dict[str, str]:
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "RNN_Sinusoid_Experiment_Leaderboard.csv"
    json_path = results_dir / "RNN_Sinusoid_Experiment_Leaderboard.json"
    if not rows:
        csv_path.write_text("")
        json_path.write_text(json.dumps({"rows": [], "winner": None}, indent=2))
        return {"csv": str(csv_path), "json": str(json_path)}
    winner = choose_winner(rows)
    fieldnames = sorted({key for row in rows for key in row.keys()} | {"winner"})
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "winner": bool(winner and row["job_id"] == winner["job_id"])})
    json_path.write_text(json.dumps({"rows": rows, "winner": winner}, indent=2))
    return {"csv": str(csv_path), "json": str(json_path)}

def build_run_status_rows(payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for payload in payloads:
        error = payload.get("error") or {}
        rows.append({
            "job_id": payload["job_id"],
            "job_name": payload["job_name"],
            "job_type": payload["job_type"],
            "family": payload["family"],
            "status": payload.get("status", "completed"),
            "gpu_id": payload.get("gpu_id"),
            "artifact_name": payload.get("artifact_name"),
            "artifact_path": payload.get("artifact_path"),
            "historical_baseline": payload.get("historical_baseline", False),
            "reused_existing": (payload.get("training") or {}).get("reused_existing") if payload.get("training") else True,
            "error_phase": error.get("phase"),
            "error_message": error.get("message"),
            "return_code": error.get("return_code"),
            "log_path": error.get("log_path"),
        })
    return rows

def write_run_status(payloads: List[Dict[str, Any]], results_dir: Path) -> Dict[str, str]:
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "RNN_Sinusoid_Experiment_Run_Status.csv"
    json_path = results_dir / "RNN_Sinusoid_Experiment_Run_Status.json"
    rows = build_run_status_rows(payloads)
    if not rows:
        csv_path.write_text("")
        json_path.write_text(json.dumps({"rows": []}, indent=2))
        return {"csv": str(csv_path), "json": str(json_path)}
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    json_path.write_text(json.dumps({"rows": rows}, indent=2))
    return {"csv": str(csv_path), "json": str(json_path)}

def build_architecture_rows(leaderboard_rows: List[Dict[str, Any]], payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows_by_family: Dict[str, List[Dict[str, Any]]] = {}
    failed_counts: Dict[str, int] = {}
    for row in leaderboard_rows:
        rows_by_family.setdefault(row["family"], []).append(row)
    for payload in payloads:
        if payload.get("status", "completed") == "completed":
            continue
        family = payload["family"]
        failed_counts[family] = failed_counts.get(family, 0) + 1
    global_winner = choose_winner(leaderboard_rows)
    families = sorted(set(rows_by_family) | set(failed_counts))
    architecture_rows = []
    for family in families:
        family_rows = rows_by_family.get(family, [])
        best_row = choose_winner(family_rows)
        baseline_row = next((row for row in family_rows if row["job_type"] == "baseline_eval" and not row["historical_baseline"]), None)
        architecture_rows.append({
            "family": family,
            "completed_jobs": len(family_rows),
            "failed_jobs": failed_counts.get(family, 0),
            "best_job_id": best_row["job_id"] if best_row else None,
            "best_job_name": best_row["job_name"] if best_row else None,
            "best_artifact_name": best_row["artifact_name"] if best_row else None,
            "best_admissible": best_row["admissible"] if best_row else None,
            "best_mean_KDR_AliceBob": best_row["mean_KDR_AliceBob"] if best_row else None,
            "best_mean_KDR_BobEve": best_row["mean_KDR_BobEve"] if best_row else None,
            "best_mean_KDR_AliceEve": best_row["mean_KDR_AliceEve"] if best_row else None,
            "best_mean_ones_fraction": best_row["mean_ones_fraction"] if best_row else None,
            "best_min_eve_gap": best_row["min_eve_gap"] if best_row else None,
            "baseline_job_id": baseline_row["job_id"] if baseline_row else None,
            "baseline_artifact_name": baseline_row["artifact_name"] if baseline_row else None,
            "delta_vs_baseline_mean_KDR_AliceBob": (best_row["mean_KDR_AliceBob"] - baseline_row["mean_KDR_AliceBob"]) if best_row and baseline_row else None,
            "delta_vs_baseline_min_eve_gap": (best_row["min_eve_gap"] - baseline_row["min_eve_gap"]) if best_row and baseline_row else None,
            "delta_vs_global_winner_mean_KDR_AliceBob": (best_row["mean_KDR_AliceBob"] - global_winner["mean_KDR_AliceBob"]) if best_row and global_winner else None,
            "delta_vs_global_winner_min_eve_gap": (best_row["min_eve_gap"] - global_winner["min_eve_gap"]) if best_row and global_winner else None,
            "winner": bool(best_row and global_winner and best_row["job_id"] == global_winner["job_id"]),
        })
    return architecture_rows

def write_architecture_comparison(leaderboard_rows: List[Dict[str, Any]], payloads: List[Dict[str, Any]], results_dir: Path) -> Dict[str, str]:
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "RNN_Sinusoid_Architecture_Comparison.csv"
    json_path = results_dir / "RNN_Sinusoid_Architecture_Comparison.json"
    architecture_rows = build_architecture_rows(leaderboard_rows, payloads)
    failed_rows = build_run_status_rows([payload for payload in payloads if payload.get("status", "completed") != "completed"])
    if architecture_rows:
        fieldnames = sorted({key for row in architecture_rows for key in row.keys()})
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in architecture_rows:
                writer.writerow(row)
    else:
        csv_path.write_text("")
    json_path.write_text(json.dumps({
        "global_winner": choose_winner(leaderboard_rows),
        "architectures": architecture_rows,
        "failed_jobs": failed_rows,
    }, indent=2))
    return {"csv": str(csv_path), "json": str(json_path)}

def scheduler_main(args: argparse.Namespace) -> int:
    ensure_expected_runtime(args.python_bin)
    home_dir = Path(args.home_dir)
    results_dir = home_dir / "Results"
    runs_dir = results_dir / "ExperimentRuns"
    runs_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args)
    print(f"Planned jobs: {len(jobs)}")
    discovered = discover_gpus()
    requested_gpu_ids, invalid_requested_gpu_ids = validate_requested_gpu_ids(
        discovered,
        parse_gpu_ids(args.gpu_ids),
    )
    print("Discovered GPUs:")
    for gpu in discovered:
        print(f"  GPU {gpu['index']}: free={gpu['memory_free_mib']} MiB util={gpu['utilization_pct']}% ({gpu['name']})")
    if invalid_requested_gpu_ids:
        print(f"[warn] Ignoring unavailable requested GPU ids: {invalid_requested_gpu_ids}")
    if args.gpu_ids and not requested_gpu_ids:
        raise SystemExit("None of the requested GPU ids are available on this host.")
    if args.dry_run:
        if requested_gpu_ids:
            print(f"Requested GPUs: {requested_gpu_ids}")
        for job in jobs:
            print(f"  - {job['job_name']} [{job['job_type']}] family={job['family']}")
        return 0

    pending_jobs = list(jobs)
    running: List[Dict[str, Any]] = []
    job_payloads: List[Dict[str, Any]] = []
    max_parallel = max(1, args.max_parallel)

    while pending_jobs or running:
        busy_gpu_ids = {record['gpu_id'] for record in running}
        available_gpu_ids = eligible_gpu_ids(
            discover_gpus(),
            requested_gpu_ids,
            busy_gpu_ids,
            args.min_free_memory_mib,
            args.max_utilization,
        )
        while pending_jobs and available_gpu_ids and len(running) < max_parallel:
            gpu_id = available_gpu_ids.pop(0)
            job = pending_jobs.pop(0)
            record = launch_worker(job, gpu_id, runs_dir, args.python_bin)
            print(f"[launch] {job['job_name']} on GPU {gpu_id} -> {record['log_path']}")
            running.append(record)

        time.sleep(args.refresh_seconds)
        still_running = []
        for record in running:
            proc = record['process']
            return_code = proc.poll()
            if return_code is None:
                still_running.append(record)
                continue
            record['log_handle'].close()
            job = record['job']
            result_path = Path(job['result_path'])
            payload = load_result_payload(result_path) if result_path.exists() else None
            if payload is None and return_code != 0:
                payload = build_failure_payload(
                    job,
                    phase="scheduler",
                    error_message=f"Worker exited with code {return_code} before writing a result payload.",
                    return_code=return_code,
                    log_path=record['log_path'],
                )
                result_path.write_text(json.dumps(payload, indent=2))
            if payload is not None:
                job_payloads.append(payload)
                if payload.get("status", "completed") == "completed":
                    print(f"[done] {job['job_name']} completed on GPU {record['gpu_id']}")
                else:
                    error_message = (payload.get("error") or {}).get("message") or f"See {record['log_path']}"
                    print(f"[fail] {job['job_name']} exited with code {return_code}. {error_message}")
            else:
                print(f"[fail] {job['job_name']} exited with code {return_code}. See {record['log_path']}")
        running = still_running
        if pending_jobs and not running and not available_gpu_ids:
            print("[wait] No eligible GPUs yet; waiting for the next refresh cycle.")

    rows = build_leaderboard_rows(job_payloads)
    leaderboard_outputs = write_leaderboard(rows, results_dir)
    status_outputs = write_run_status(job_payloads, results_dir)
    comparison_outputs = write_architecture_comparison(rows, job_payloads, results_dir)
    winner = choose_winner(rows)
    failed_payloads = [payload for payload in job_payloads if payload.get("status", "completed") != "completed"]
    print(f"Leaderboard CSV: {leaderboard_outputs['csv']}")
    print(f"Leaderboard JSON: {leaderboard_outputs['json']}")
    print(f"Run status CSV: {status_outputs['csv']}")
    print(f"Run status JSON: {status_outputs['json']}")
    print(f"Architecture comparison CSV: {comparison_outputs['csv']}")
    print(f"Architecture comparison JSON: {comparison_outputs['json']}")
    if winner:
        print(f"Winner: {winner['artifact_name']} (family={winner['family']})")
    if failed_payloads:
        print(f"Failed jobs: {len(failed_payloads)}")
    return 0 if not failed_payloads else 1


def worker_main(job_json: str) -> int:
    job = json.loads(Path(job_json).read_text())
    run_worker(job)
    return 0


def main() -> int:
    args = parse_args()
    if args.worker_job_json:
        return worker_main(args.worker_job_json)
    return scheduler_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
