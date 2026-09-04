# Result Store Design - PHASE 4

**Goal**: Make runs accumulate so users don't re-run identical configurations

---

## Design: JSON-Based Result Index

**Choice**: JSON index over SQLite for portability and inspectability

**Location**: `~/.throttle/results/index.json` (XDG_DATA_HOME compliant)

### Index Structure

```json
{
  "version": 1,
  "last_updated": "2026-08-29T10:30:00Z",
  "results": [
    {
      "id": "golden_20260829_103000_abc123",
      "config_key": "Qwen-Qwen2.5-8B-Instruct_A100-80GB-PCIe_vllm_max-num-seqs-1-to-8_standard-workload",
      "artifact_type": "throttle_golden_live_comparison",
      "decision_eligible": true,
      "decision_state": "supported",
      "created_at": "2026-08-29T10:30:00Z",
      "file_path": "~/.throttle/results/golden_20260829_103000_abc123.json",
      "metadata": {
        "model": "Qwen/Qwen2.5-8B-Instruct",
        "gpu": "A100 80GB PCIe",
        "backend": "vllm",
        "baseline_config": {"max_num_seqs": 1},
        "candidate_config": {"max_num_seqs": 8},
        "workload_type": "standard",
        "throughput_delta_percent": {
          "estimate": 217.8,
          "low": 189.5,
          "high": 246.2
        },
        "gpu_hourly_rate": 1.39
      }
    },
    {
      "id": "benchmark_20260829_120000_def456",
      "config_key": "Meta-Llama-3.1-8B-Instruct_RTX-4090_ollama_max-num-seqs-4_custom-workload",
      "artifact_type": "throttle_benchmark_report",
      "decision_eligible": false,
      "created_at": "2026-08-29T12:00:00Z",
      "file_path": "~/.throttle/results/benchmark_20260829_120000_def456.json",
      "metadata": {
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "gpu": "RTX 4090",
        "backend": "ollama",
        "config": {"max_num_seqs": 4},
        "workload_type": "custom",
        "ineligible_reason": "non_counterbalanced"
      }
    }
  ]
}
```

### Config Key Format

Canonical key for detecting duplicate configurations:

```
{model_family}_{gpu_normalized}_{backend}_{param_variation}_{workload_fingerprint}
```

**Examples**:
- `Qwen2.5-8B_A100-80GB_vllm_max-num-seqs-1-to-8_std-workload-sha256-abc123`
- `Llama-3.1-8B_RTX-4090_ollama_gpu-mem-util-0.85-to-0.95_custom-sha256-def456`

**Normalization rules**:
- Model: Extract family name (Qwen2.5, Llama-3.1, etc.)
- GPU: Normalize vendor + tier + VRAM (A100-80GB, RTX-4090-24GB)
- Backend: Lowercase (vllm, ollama, sglang)
- Parameters: Sorted keys with baseline→candidate format
- Workload: Type + SHA256 of workload definition (first 8 chars)

---

## Integration Points

### 1. Pre-Run Detection

**Location**: Before `throttle golden` or `throttle benchmark` starts

**Workflow**:
```python
def check_prior_result(config: Dict) -> Optional[PriorResult]:
    """Check if this exact config was already run."""
    index = load_result_index()
    config_key = generate_config_key(config)

    matches = [r for r in index["results"] if r["config_key"] == config_key]
    if not matches:
        return None

    # Return most recent decision-eligible result, or most recent any result
    decision_eligible = [m for m in matches if m["decision_eligible"]]
    if decision_eligible:
        return max(decision_eligible, key=lambda x: x["created_at"])
    return max(matches, key=lambda x: x["created_at"])
```

**User experience**:
```bash
$ throttle golden --endpoint-url https://... --model Qwen/Qwen2.5-8B \
    --baseline-max-num-seqs 1 --candidate-max-num-seqs 8

⚠ PRIOR RESULT FOUND
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Config: Qwen2.5-8B on A100 80GB (vLLM)
  Baseline: max_num_seqs=1
  Candidate: max_num_seqs=8

Previous run: 2026-08-29 10:30 UTC (2 days ago)
  Decision-eligible: ✓ Yes
  Outcome: Candidate +217.8% throughput (CI: +189.5% to +246.2%)
  Cost: $1.04 (45 minutes @ $1.39/hr)

Result file: ~/.throttle/results/golden_20260829_103000_abc123.json

[1] Use prior result (no GPU time spent)
[2] Re-run anyway (fresh measurement)
[3] Cancel

Choice: _
```

### 2. Post-Run Storage

**Location**: After successful `throttle golden` or `throttle benchmark`

**Workflow**:
```python
def store_result(result_file: Path, config: Dict, metadata: Dict) -> None:
    """Add result to index after successful run."""
    index = load_result_index()

    result_id = generate_result_id(result_file)
    config_key = generate_config_key(config)

    index["results"].append({
        "id": result_id,
        "config_key": config_key,
        "artifact_type": metadata["artifact_type"],
        "decision_eligible": metadata["decision_eligible"],
        "created_at": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        "file_path": str(result_file.resolve()),
        "metadata": metadata,
    })

    index["last_updated"] = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    save_result_index(index)
```

### 3. `throttle recommend` Command

**New command**: Suggest optimal starting config for a described workload

**Usage**:
```bash
$ throttle recommend --model Qwen/Qwen2.5-8B-Instruct --gpu "A100 80GB" --backend vllm

Throttle Configuration Recommendations
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model: Qwen/Qwen2.5-8B-Instruct
GPU: A100 80GB PCIe
Backend: vLLM

PRIOR RESULTS:
  ✓ max_num_seqs: 1 → 8 (+217.8% throughput) [2 days ago]
    File: ~/.throttle/results/golden_20260829_103000_abc123.json

RECOMMENDED STARTING CONFIG:
  --max-num-seqs 8
  --gpu-memory-utilization 0.95  # Based on A100 80GB capacity

NEXT EXPLORATION:
  Try max_num_seqs: 8 → 16 to find saturation point
  Estimated cost: $1.04 (45 min @ $1.39/hr)

To apply recommendation:
  throttle benchmark --endpoint-url YOUR_URL \\
    --model Qwen/Qwen2.5-8B-Instruct \\
    --max-num-seqs 8 \\
    --gpu-memory-utilization 0.95
```

---

## Implementation Phases

### Phase 4a: Storage Infrastructure (Foundational)

**Files to create**:
- `src/throttle/result_store.py` - Core storage logic
- `~/.throttle/results/` - User data directory
- `~/.throttle/results/index.json` - Index file

**Functions**:
- `load_result_index() -> Dict`
- `save_result_index(index: Dict) -> None`
- `generate_config_key(config: Dict) -> str`
- `generate_result_id(result_file: Path) -> str`
- `add_result(result_file: Path, config: Dict, metadata: Dict) -> None`
- `find_matching_results(config: Dict) -> List[Dict]`

### Phase 4b: CLI Integration (User-facing)

**Modify**:
- `src/throttle/cli.py:_handle_golden()` - Add pre-run check, post-run storage
- `src/throttle/cli.py:_handle_benchmark()` - Same

**Add**:
- `src/throttle/cli.py:_handle_recommend()` - New command handler

### Phase 4c: Tests (Validation)

**Files**:
- `tests/test_result_store.py` - Unit tests for storage
- `tests/test_recommend.py` - Test recommend command
- `tests/test_golden_with_store.py` - Integration test

---

## Portability & Inspectability

### Portable
- **Pure JSON**: No database dependencies
- **Relative paths**: Store paths relative to `~/.throttle/results/`
- **Version field**: Allow future schema migrations
- **No absolute hostnames**: Config keys use normalized GPU/model names, not endpoint URLs

### Inspectable
- **Human-readable**: JSON with 2-space indentation
- **grep-able**: `cat ~/.throttle/results/index.json | jq '.results[] | select(.decision_eligible == true)'`
- **git-friendly**: Can commit index.json to track experiment history

### Export/Import
```bash
# Export index to share with team
throttle export-results --output team_results.json

# Import results from teammate
throttle import-results --input teammate_results.json --merge

# Clear all results
throttle clear-results --confirm
```

---

## Definition of Done (PHASE 4)

- [x] Design document created (this file)
- [ ] `result_store.py` module implemented
- [ ] Pre-run detection added to `golden` and `benchmark` commands
- [ ] Post-run storage added to `golden` and `benchmark` commands
- [ ] `throttle recommend` command implemented
- [ ] Tests prove: second run of same config reads first from store
- [ ] Manual test: Run golden twice, second run shows prior result prompt
- [ ] Manual test: `throttle recommend` outputs sensible starting config

---

## Future Enhancements

1. **Cloud sync**: `throttle sync --remote s3://my-results-bucket`
2. **Team sharing**: `throttle share-result <result-id> --with user@example.com`
3. **Web UI**: `throttle serve-results` launches local dashboard
4. **Auto-exploration**: `throttle explore --model X --gpu Y --auto` tries config variations intelligently
5. **Cost tracking**: Aggregate total GPU spend across all stored results
