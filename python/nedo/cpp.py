from __future__ import annotations
from pathlib import Path
from ._hub import parse_hf_source, select_gguf, select_tokenizer_sidecar

try:
    from ._core import Model as _NativeModel, ModelConfig, SchemaReport, GenerationConfig
except ImportError as exc:
    _NativeModel = ModelConfig = SchemaReport = GenerationConfig = None
    _CORE_IMPORT_ERROR = exc
else:
    _CORE_IMPORT_ERROR = None


def _require_core() -> None:
    if _CORE_IMPORT_ERROR is not None:
        raise RuntimeError("nedo.cpp native extension is not installed. Run `pip install .` or install a wheel.") from _CORE_IMPORT_ERROR


def _surface_tokenizer(path: Path):
    try:
        from nedotokenizer import SurfaceTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Exact NDSRF004 tokenization requires Ethosoft NedoTokenizer. "
            "Install it from https://github.com/ethosoftai/NedoTokenizer before loading NedoLM."
        ) from exc
    return SurfaceTokenizer(path.read_bytes())


class Model:
    """Public NedoLM model wrapper.

    Transformer inference stays in the C++ core. NDSRF004 text segmentation is
    delegated to the canonical Ethosoft NedoTokenizer implementation so the
    public Python API never uses an approximate morphology scanner.
    """

    def __init__(self, model_path: str | Path, tokenizer_path: str | Path):
        _require_core()
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path)
        self._native = _NativeModel(self.model_path)
        self._tokenizer = _surface_tokenizer(self.tokenizer_path)

    @property
    def config(self):
        return self._native.config

    @property
    def schema(self):
        return self._native.schema

    def summary(self) -> str:
        return self._native.summary() + "\ntokenizer: canonical NDSRF004 (Ethosoft/NedoTokenizer)"

    def tokenize(self, text: str, add_bos: bool = False) -> list[int]:
        ids = [int(value) for value in self._tokenizer.encode_ids(text.encode("utf-8"))]
        if add_bos and (not ids or ids[0] != 1):
            ids.insert(0, 1)
        return ids

    def detokenize(self, ids: list[int]) -> str:
        data = self._tokenizer.decode_ids([int(value) for value in ids])
        return bytes(data).decode("utf-8", errors="replace")

    def generate_ids(self, prompt_ids: list[int], config=None) -> list[int]:
        if config is None:
            config = GenerationConfig()
        return [int(value) for value in self._native.generate_ids([int(v) for v in prompt_ids], config)]

    def generate(self, prompt: str, config=None) -> str:
        if config is None:
            config = GenerationConfig()
        prompt_ids = self.tokenize(prompt, add_bos=False)
        generated = self.generate_ids(prompt_ids, config)
        return self.detokenize(generated)

    @property
    def native(self):
        """Low-level C++ model. Prefer the wrapper methods for text I/O."""
        return self._native


def _hf_download_bundle(repo: str, *, filename: str | None = None, quantization: str | None = None, cache_dir: str | Path | None = None, revision: str | None = None) -> tuple[Path, Path]:
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
    sidecar = select_tokenizer_sidecar(files, model_filename)
    if sidecar is None:
        raise FileNotFoundError("The selected NedoLM GGUF has no surface-vocab*.bin tokenizer sidecar")
    model_path = Path(hf_hub_download(source.repo_id, model_filename, repo_type="model", revision=rev, cache_dir=cache_dir))
    tokenizer_path = Path(hf_hub_download(source.repo_id, sidecar, repo_type="model", revision=rev, cache_dir=cache_dir))
    return model_path, tokenizer_path


def hf_download(repo: str, *, filename: str | None = None, quantization: str | None = None, cache_dir: str | Path | None = None, revision: str | None = None) -> Path:
    model_path, _ = _hf_download_bundle(repo, filename=filename, quantization=quantization, cache_dir=cache_dir, revision=revision)
    return model_path


def _local_tokenizer_sidecar(model_path: Path) -> Path:
    files = [item.name for item in model_path.parent.iterdir() if item.is_file()]
    sidecar = select_tokenizer_sidecar(files, model_path.name)
    if sidecar is None:
        raise FileNotFoundError(
            f"No surface-vocab*.bin tokenizer sidecar found next to local model {model_path}"
        )
    return model_path.parent / sidecar


def import_llm(model: str | Path, *, filename: str | None = None, quantization: str | None = None, cache_dir: str | Path | None = None, revision: str | None = None):
    _require_core()
    raw = str(model)
    p = Path(raw).expanduser()
    if p.suffix.lower() == ".gguf" and p.exists():
        model_path = p
        tokenizer_path = _local_tokenizer_sidecar(model_path)
    else:
        model_path, tokenizer_path = _hf_download_bundle(raw, filename=filename, quantization=quantization, cache_dir=cache_dir, revision=revision)
    return Model(model_path, tokenizer_path)


load = import_llm
__all__ = ["Model", "ModelConfig", "SchemaReport", "GenerationConfig", "hf_download", "import_llm", "load"]
