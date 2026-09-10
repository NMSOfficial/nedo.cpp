from __future__ import annotations
from pathlib import Path
from ._hub import parse_hf_source, select_gguf, select_tokenizer_sidecar

try:
    from ._core import Model, ModelConfig, SchemaReport, GenerationConfig
except ImportError as exc:
    Model = ModelConfig = SchemaReport = GenerationConfig = None
    _CORE_IMPORT_ERROR = exc
else:
    _CORE_IMPORT_ERROR = None


def _require_core() -> None:
    if _CORE_IMPORT_ERROR is not None:
        raise RuntimeError("nedo.cpp native extension is not installed. Run `pip install .` or install a wheel.") from _CORE_IMPORT_ERROR


def hf_download(repo: str, *, filename: str | None = None, quantization: str | None = None, cache_dir: str | Path | None = None, revision: str | None = None) -> Path:
    source = parse_hf_source(repo)
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise RuntimeError("`huggingface_hub` is required; reinstall `nedo-cpp` to restore dependencies.") from exc
    rev = revision or source.revision
    requested_filename = filename or source.filename
    api = HfApi()
    files = list(api.list_repo_files(source.repo_id, repo_type="model", revision=rev))
    model_filename = select_gguf(files, filename=requested_filename, quantization=quantization)
    model_path = Path(hf_hub_download(source.repo_id, model_filename, repo_type="model", revision=rev, cache_dir=cache_dir))
    sidecar = select_tokenizer_sidecar(files, model_filename)
    if sidecar:
        hf_hub_download(source.repo_id, sidecar, repo_type="model", revision=rev, cache_dir=cache_dir)
    return model_path


def import_llm(model: str | Path, *, filename: str | None = None, quantization: str | None = None, cache_dir: str | Path | None = None, revision: str | None = None):
    _require_core()
    raw = str(model)
    p = Path(raw).expanduser()
    if p.suffix.lower() == ".gguf" and p.exists():
        model_path = p
    else:
        model_path = hf_download(raw, filename=filename, quantization=quantization, cache_dir=cache_dir, revision=revision)
    return Model(model_path)


load = import_llm
__all__ = ["Model", "ModelConfig", "SchemaReport", "GenerationConfig", "hf_download", "import_llm", "load"]
