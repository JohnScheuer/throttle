# Known Issues and Findings

## Wheel Parity Job - Cold Cache Failure (UNFIXED)

**Status**: Known failure on cold cache, masked by warm cache in CI

**Problem**: The wheel parity job will fail under PYTHONWARNINGS=error when HuggingFace Hub cache is cold due to deprecated hf-xet API usage.

**Evidence**:
- Run 32702491665 (cold cache): FAILED with `DeprecationWarning: hf_xet.download_files() is deprecated`
- Run 32704183533 (warm cache): PASSED - cache hit prevented network download

**Cache Configuration**:
```yaml
- name: Cache HuggingFace Hub directory
  uses: actions/cache@1bd1e32a3bdc45362d1e726936510720a7c30a57
  with:
    path: ~/.cache/huggingface/hub
    key: hf-hub-Linux-sentence-transformers-all-MiniLM-L6-v2
```

**Root Cause**: huggingface-hub 0.36.2 uses deprecated hf_xet.download_files() API internally when downloading model files. This triggers DeprecationWarning which fails under warning-strict mode.

**When Failure Occurs**:
- Cold cache (new cache key, first run, or cache expiration)
- Fork repositories (different cache namespace)
- Cache invalidation or manual cache clear

**Attempted Fixes**:
- Option a (pin/update huggingface-hub): No version without hf-xet deprecation exists
- Option b (env var to disable hf-xet): No such environment variable exists

**Workaround** (NOT IMPLEMENTED):
Add scoped filterwarnings to pyproject.toml:
```toml
[tool.pytest.ini_options]
filterwarnings = [
    "ignore:hf_xet\\.download_files\\(\\) is deprecated:DeprecationWarning:huggingface_hub\\.file_download",
]
```

**Decision**: Documented as known cold cache failure. Fix requires either:
1. huggingface-hub upstream fix (remove hf-xet dependency)
2. Implement scoped filterwarnings (relaxes warning-strict for this specific case)

**Impact**: Low - CI cache hit rate is high, forks can add filterwarnings if needed
