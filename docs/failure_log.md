# Failure Log — Recovery Decision Engine

This document provides a consolidated audit of all development errors, API quirks, statistical sampling challenges, and resolution engineering across all phases of the project.

---

## 1. Environment & Binary Compatibility Issues

### Issue 1.1: System Python NumPy / Pandas Binary Mismatch
- **Symptom**: `ValueError: numpy.dtype size changed, may indicate binary incompatibility` when importing pandas on system Python 3.11.
- **Root Cause**: System-level global site-packages contained mismatched C-extension headers for `numpy` and `pandas`.
- **Resolution**: Created a clean isolated virtual environment (`./venv`) and managed matched dependencies via [`requirements.txt`](file:///Users/prashantshukla/Desktop/Recovery%20Decision%20Engine/requirements.txt).

---

## 2. Phase 1 — Data Simulator Issues

### Issue 2.1: NumPy Random Generator Method Missing
- **Symptom**: `AttributeError: 'numpy.random._generator.Generator' object has no attribute 'hex'` during `event_id` generation.
- **Root Cause**: Assumed `.hex()` method existed on NumPy's `default_rng()`.
- **Resolution**: Replaced with `f"{rng.integers(0, 0xFFFFFFFF):08x}"` in [`src/simulation/generator.py`](file:///Users/prashantshukla/Desktop/Recovery%20Decision%20Engine/src/simulation/generator.py) for reproducible deterministic hex string suffix generation.

### Issue 2.2: CLI Argument Parser Hyphen-to-Underscore Mapping
- **Symptom**: `AttributeError: 'Namespace' object has no attribute 'output'` when calling `cli.py --output-dir data`.
- **Root Cause**: `argparse` converts hyphens (`--output-dir`) to underscores (`args.output_dir`) in the parsed Namespace object.
- **Resolution**: Updated `cli.py` reference from `args.output-dir` to `args.output_dir`.

---

## 3. Phase 2 — Probabilistic Modeling & PyMC Issues

### Issue 3.1: Preprocessing Vector Shape Mismatch for Single Inferences
- **Symptom**: `ValueError: matmul shapes (5,) and (5, 1) not aligned` when scoring a single `FailureContext`.
- **Root Cause**: `FeaturePreprocessor.transform_df()` produced 2D arrays `(N, K)`, whereas single-dictionary transformation returned 1D arrays `(K,)`, breaking dot product matrix multiplication against PyMC posterior samples `(draws, K)`.
- **Resolution**: Built [`transform_single()`](file:///Users/prashantshukla/Desktop/Recovery%20Decision%20Engine/src/modeling/bayesian_model.py#L42) in `FeaturePreprocessor` to explicitly return 2D matrix shape `(1, K)`.

---

## 4. Phase 4 — Agent Orchestration & API Integration Issues

### Issue 4.1: Missing Anthropic API Credentials in Local Environment
- **Symptom**: Pipeline failure if `ANTHROPIC_API_KEY` environment variable is not present during LLM context parsing or customer message generation.
- **Root Cause**: Hard dependency on external HTTP API calls.
- **Resolution**: Implemented high-fidelity deterministic rule-based fallbacks in [`src/llm/context_builder.py`](file:///Users/prashantshukla/Desktop/Recovery%20Decision%20Engine/src/llm/context_builder.py) and [`src/llm/message_generator.py`](file:///Users/prashantshukla/Desktop/Recovery%20Decision%20Engine/src/llm/message_generator.py). The pipeline degrades gracefully without crashing.

### Issue 4.2: Non-API Interventions in Razorpay REST Test Mode
- **Symptom**: Actions like `voice_call` or `escalate_human` do not have live REST endpoints in Razorpay's test-mode API.
- **Root Cause**: Razorpay REST API test mode supports payments and notifications, but not human representative assignment.
- **Resolution**: Built simulated test handlers in [`src/razorpay_client/client.py`](file:///Users/prashantshukla/Desktop/Recovery%20Decision%20Engine/src/razorpay_client/client.py) that assign test transaction IDs, return acknowledgment payloads, and log complete audit entries to SQLite.
