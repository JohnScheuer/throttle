#!/usr/bin/env python3
"""
Golden Protocol Matrix Runner

Runs the golden protocol (six-position counterbalanced B1/C1/B2/C2/B3/C3 testing)
across a configuration matrix for replication and validation.

Usage:
    ./scripts/run_golden_matrix.py --matrix matrix.yaml [--resume]

Matrix file format (YAML):
    cells:
      - name: "A100-Qwen-8B-max_num_seqs"
        endpoint: "https://runpod-endpoint.com/v1"
        api_key_env: "RUNPOD_API_KEY"
        model: "Qwen/Qwen2.5-8B-Instruct"
        gpu: "A100 80GB PCIe"
        gpu_hourly_rate: 1.39
        baseline_config:
          max_num_seqs: 1
        candidate_config:
          max_num_seqs: 8
        estimated_duration_minutes: 45

      - name: "RTX4090-Qwen-8B-max_num_seqs"
        ...
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:
    print("Error: pyyaml required for matrix file parsing")
    print("Install with: pip install pyyaml")
    sys.exit(1)


def load_matrix(matrix_file: Path) -> Dict[str, Any]:
    """Load and validate matrix configuration."""
    with open(matrix_file) as f:
        matrix = yaml.safe_load(f)

    if "cells" not in matrix:
        raise ValueError("Matrix file must contain 'cells' key")

    for i, cell in enumerate(matrix["cells"]):
        required = ["name", "endpoint", "model", "gpu", "gpu_hourly_rate",
                    "baseline_config", "candidate_config"]
        missing = [k for k in required if k not in cell]
        if missing:
            raise ValueError(f"Cell {i} missing required keys: {missing}")

    return matrix


def estimate_cost(cell: Dict[str, Any]) -> Dict[str, float]:
    """Estimate RunPod cost for a golden protocol run."""
    duration_minutes = cell.get("estimated_duration_minutes", 60)
    hourly_rate = cell["gpu_hourly_rate"]

    cost_usd = (duration_minutes / 60) * hourly_rate

    return {
        "estimated_duration_minutes": duration_minutes,
        "estimated_cost_usd": round(cost_usd, 4),
        "gpu_hourly_rate_usd": hourly_rate,
    }


def build_golden_command(cell: Dict[str, Any], output_dir: Path) -> List[str]:
    """Build throttle golden command for a matrix cell."""
    baseline = cell["baseline_config"]
    candidate = cell["candidate_config"]

    # Determine the parameter being varied
    varied_params = set(candidate.keys()) & set(baseline.keys())
    if not varied_params:
        raise ValueError(f"Cell {cell['name']}: baseline and candidate must share at least one parameter")

    # Build command
    cmd = [
        "throttle", "golden",
        "--endpoint-url", cell["endpoint"],
        "--model", cell["model"],
        "--gpu-rate-per-hour", str(cell["gpu_hourly_rate"]),
    ]

    # Add API key if specified
    if "api_key_env" in cell:
        api_key = os.getenv(cell["api_key_env"])
        if not api_key:
            raise ValueError(f"Environment variable {cell['api_key_env']} not set")
        cmd.extend(["--api-key", api_key])

    # Add baseline and candidate configs
    for param, value in baseline.items():
        cmd.extend([f"--baseline-{param.replace('_', '-')}", str(value)])

    for param, value in candidate.items():
        cmd.extend([f"--candidate-{param.replace('_', '-')}", str(value)])

    # Add output path
    output_file = output_dir / f"{cell['name']}_golden.json"
    cmd.extend(["--output", str(output_file)])

    return cmd


def run_cell(cell: Dict[str, Any], output_dir: Path, resume: bool = False) -> Dict[str, Any]:
    """Run golden protocol for a single matrix cell."""
    cell_name = cell["name"]
    output_file = output_dir / f"{cell_name}_golden.json"

    # Check if already completed (resume support)
    if resume and output_file.exists():
        try:
            with open(output_file) as f:
                result = json.load(f)
                if result.get("decision_eligible"):
                    print(f"  [RESUME] {cell_name}: Already complete (decision_eligible=true)")
                    return {
                        "cell_name": cell_name,
                        "status": "resumed",
                        "output_file": str(output_file),
                        "decision_eligible": True,
                        "result": result,
                    }
        except (json.JSONDecodeError, KeyError):
            pass  # File corrupt or incomplete, re-run

    # Build and run command
    try:
        cmd = build_golden_command(cell, output_dir)
        print(f"  [RUN] {cell_name}")
        print(f"        {' '.join(cmd)}")

        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=cell.get("timeout_seconds", 7200),  # 2 hour default timeout
        )
        elapsed_minutes = (time.time() - start_time) / 60

        if result.returncode != 0:
            return {
                "cell_name": cell_name,
                "status": "failed",
                "exit_code": result.returncode,
                "stderr": result.stderr[-1000:],  # Last 1000 chars
                "elapsed_minutes": round(elapsed_minutes, 2),
            }

        # Load and validate result
        with open(output_file) as f:
            golden_result = json.load(f)

        return {
            "cell_name": cell_name,
            "status": "success",
            "output_file": str(output_file),
            "decision_eligible": golden_result.get("decision_eligible", False),
            "decision_state": golden_result.get("decision_state"),
            "elapsed_minutes": round(elapsed_minutes, 2),
            "result": golden_result,
        }

    except subprocess.TimeoutExpired:
        return {
            "cell_name": cell_name,
            "status": "timeout",
            "timeout_seconds": cell.get("timeout_seconds", 7200),
        }
    except Exception as e:
        return {
            "cell_name": cell_name,
            "status": "error",
            "error": str(e),
        }


def emit_summary_table(results: List[Dict[str, Any]], matrix: Dict[str, Any]) -> None:
    """Print summary table of matrix run results."""
    print("\n" + "=" * 100)
    print("GOLDEN PROTOCOL MATRIX - SUMMARY")
    print("=" * 100)
    print()

    # Table header
    print(f"{'Cell Name':<40} {'Status':<12} {'Decision':<12} {'Duration':<12} {'Cost Estimate':<15}")
    print("-" * 100)

    # Table rows
    total_cost = 0.0
    decision_eligible_count = 0

    for result in results:
        cell_name = result["cell_name"]
        status = result["status"]

        # Find corresponding cell for cost estimate
        cell = next((c for c in matrix["cells"] if c["name"] == cell_name), None)
        cost_est = estimate_cost(cell) if cell else {"estimated_cost_usd": 0.0}

        decision = "✓ eligible" if result.get("decision_eligible") else "✗ ineligible"
        if status != "success":
            decision = "N/A"

        duration = f"{result.get('elapsed_minutes', 0):.1f} min"
        cost = f"${cost_est['estimated_cost_usd']:.4f}"

        print(f"{cell_name:<40} {status:<12} {decision:<12} {duration:<12} {cost:<15}")

        if result.get("decision_eligible"):
            decision_eligible_count += 1
        total_cost += cost_est["estimated_cost_usd"]

    print("-" * 100)
    print(f"{'TOTAL':<40} {'':<12} {decision_eligible_count} eligible {'':<12} ${total_cost:.4f}")
    print()

    # Failed cells details
    failed = [r for r in results if r["status"] not in ["success", "resumed"]]
    if failed:
        print("\nFAILED CELLS (details):")
        for r in failed:
            print(f"  {r['cell_name']}: {r['status']}")
            if "error" in r:
                print(f"    Error: {r['error']}")
            if "stderr" in r:
                print(f"    Stderr: {r['stderr'][:200]}...")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Run golden protocol across configuration matrix"
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        required=True,
        help="Path to matrix YAML file",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip cells that already have decision_eligible=true results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("matrix_results"),
        help="Directory for output files (default: matrix_results/)",
    )
    args = parser.parse_args()

    # Load matrix
    print(f"Loading matrix from {args.matrix}")
    matrix = load_matrix(args.matrix)
    print(f"Found {len(matrix['cells'])} cells")
    print()

    # Create output directory
    args.output_dir.mkdir(exist_ok=True)

    # Print cost estimates
    print("COST ESTIMATES:")
    total_estimate = 0.0
    for cell in matrix["cells"]:
        cost_info = estimate_cost(cell)
        print(f"  {cell['name']}: ${cost_info['estimated_cost_usd']:.4f} "
              f"({cost_info['estimated_duration_minutes']} min @ "
              f"${cost_info['gpu_hourly_rate_usd']}/hr)")
        total_estimate += cost_info["estimated_cost_usd"]
    print(f"  TOTAL ESTIMATED: ${total_estimate:.4f}")
    print()

    # Run matrix cells
    print("RUNNING MATRIX:")
    results = []
    for i, cell in enumerate(matrix["cells"], 1):
        print(f"\n[{i}/{len(matrix['cells'])}] Processing: {cell['name']}")
        result = run_cell(cell, args.output_dir, resume=args.resume)
        results.append(result)

        # Fail loudly if cell failed
        if result["status"] not in ["success", "resumed"]:
            print(f"  [ERROR] Cell failed: {result.get('error', result.get('stderr', 'Unknown error'))}")
            print(f"  [ERROR] Matrix run stopped. Use --resume to skip completed cells.")

            # Emit partial summary
            emit_summary_table(results, matrix)
            sys.exit(1)

    # Emit final summary
    emit_summary_table(results, matrix)

    # Write summary JSON
    summary_file = args.output_dir / "matrix_summary.json"
    with open(summary_file, "w") as f:
        json.dump({
            "matrix_file": str(args.matrix),
            "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            "total_cells": len(matrix["cells"]),
            "decision_eligible_count": sum(1 for r in results if r.get("decision_eligible")),
            "results": results,
        }, f, indent=2)

    print(f"Full summary written to: {summary_file}")

    # Exit with error if any cell didn't achieve decision_eligible=true
    if not all(r.get("decision_eligible") for r in results):
        print("\nWARNING: Not all cells achieved decision_eligible=true")
        sys.exit(1)

    print("\n✓ All cells achieved decision_eligible=true")
    sys.exit(0)


if __name__ == "__main__":
    main()
