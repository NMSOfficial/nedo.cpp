# 0.1.0a5

- Make `huggingface_hub` a required dependency for one-command GitHub/PyPI installs.
- Prepare GitHub-first distribution.

# Changelog

## 0.1.0a4

- Add `nedo chat OWNER/REPO` interactive REPL.
- Keep conversation history using the Nedo SFT prompt format.
- Print model responses as raw UTF-8 text; no token IDs or JSON envelope.
- Add `/clear`, `/exit`, `/quit` commands and generation controls.

## 0.1.0a3

- Removed automatic Q4_0 -> Q8_0 -> F16 selection.
- Repository loads now show a numbered quantization prompt before downloading any GGUF model bytes.
- Explicit `quantization=` and `filename=` remain available for non-interactive scripts and deterministic deployments.
- Invalid interactive choices are rejected and re-prompted.

## 0.1.0a2

- Replaced the built-in model alias/registry API with direct Hugging Face repository IDs.
- `nedo.import_llm("owner/repo")` now resolves, downloads and loads GGUF directly.
- Full `huggingface.co/owner/repo` and model-file URLs are accepted.
- Added explicit `quantization=` and `filename=` overrides.
- Added automatic Nedo `surface-vocab*.bin` sidecar discovery/download.

## 0.1.0-alpha.1 — 2026-09-10

Initial public-runtime foundation for NedoLM GGUF.
