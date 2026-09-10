from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse, unquote
import re


@dataclass(frozen=True, slots=True)
class HubSource:
    repo_id: str
    filename: str | None = None
    revision: str = "main"


_HF_HOSTS = {"huggingface.co", "www.huggingface.co", "hf.co", "www.hf.co"}
_QUANT_ALIASES = {
    "Q4": "Q4_0",
    "Q4_0": "Q4_0",
    "Q8": "Q8_0",
    "Q8_0": "Q8_0",
    "F16": "F16",
    "FP16": "F16",
}
_SUPPORTED_QUANTIZATIONS = ("Q4_0", "Q8_0", "F16")


def parse_hf_source(value: str) -> HubSource:
    value = value.strip()
    if not value:
        raise ValueError("Empty Hugging Face model source")
    if value.startswith("hf://"):
        value = value[5:].strip("/")
    if value.startswith("http://") or value.startswith("https://"):
        parsed = urlparse(value)
        if parsed.netloc.lower() not in _HF_HOSTS:
            raise ValueError(f"Only Hugging Face URLs are supported, got host {parsed.netloc!r}")
        parts = [unquote(p) for p in parsed.path.split("/") if p]
        if len(parts) < 2:
            raise ValueError(f"Invalid Hugging Face model URL: {value!r}")
        repo_id = f"{parts[0]}/{parts[1]}"
        if len(parts) >= 5 and parts[2] in {"blob", "resolve"}:
            return HubSource(repo_id=repo_id, filename="/".join(parts[4:]), revision=parts[3])
        return HubSource(repo_id=repo_id)
    parts = value.strip("/").split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError("Expected a Hugging Face repo id like 'Ethosoft/NedoLM-0.8B-SFT-GGUF' or a huggingface.co model URL")
    return HubSource(repo_id=value.strip("/"))


def normalize_quantization(value: str) -> str:
    q = value.upper().replace("-", "_")
    try:
        return _QUANT_ALIASES[q]
    except KeyError:
        raise ValueError("quantization must be one of: Q4_0, Q8_0, F16") from None


def _quant_from_name(filename: str) -> str | None:
    name = Path(filename).name.upper()
    if re.search(r"(?:^|[-_.])Q4_0(?:[-_.]|$)", name):
        return "Q4_0"
    if re.search(r"(?:^|[-_.])Q8_0(?:[-_.]|$)", name):
        return "Q8_0"
    if re.search(r"(?:^|[-_.])F16(?:[-_.]|$)", name) or re.search(r"(?:^|[-_.])FP16(?:[-_.]|$)", name):
        return "F16"
    return None


def quantization_choices(files: list[str]) -> list[str]:
    present = {q for f in files if f.lower().endswith(".gguf") for q in [_quant_from_name(f)] if q is not None}
    return [q for q in _SUPPORTED_QUANTIZATIONS if q in present]


def prompt_quantization(choices: list[str], *, input_fn: Callable[[str], str] = input, output_fn: Callable[[str], None] = print) -> str:
    if not choices:
        raise RuntimeError("No nedo.cpp-supported GGUF quantization was found in the repository")
    output_fn("Bulunan quantization seçenekleri:")
    for i, choice in enumerate(choices, start=1):
        output_fn(f"{i}) {choice}")
    while True:
        try:
            raw = input_fn(f"Hangisini seçersiniz? [1-{len(choices)}]: ").strip()
        except (EOFError, KeyboardInterrupt) as exc:
            raise RuntimeError("Interactive quantization selection is unavailable. Pass quantization='Q4_0', 'Q8_0' or 'F16' explicitly.") from exc
        if raw.isdigit():
            index = int(raw)
            if 1 <= index <= len(choices):
                return choices[index - 1]
        output_fn(f"Geçersiz seçim. 1 ile {len(choices)} arasında bir sayı girin.")


def select_gguf(files: list[str], *, filename: str | None = None, quantization: str | None = None, input_fn: Callable[[str], str] = input, output_fn: Callable[[str], None] = print) -> str:
    ggufs = [f for f in files if f.lower().endswith(".gguf")]
    if filename is not None:
        if filename not in files:
            raise FileNotFoundError(f"{filename!r} was not found in the Hugging Face repository")
        if not filename.lower().endswith(".gguf"):
            raise ValueError(f"Selected file is not a GGUF: {filename!r}")
        return filename
    if not ggufs:
        raise FileNotFoundError("No .gguf file found in the Hugging Face repository")
    by_q: dict[str, list[str]] = {q: [] for q in _SUPPORTED_QUANTIZATIONS}
    for f in ggufs:
        fq = _quant_from_name(f)
        if fq in by_q:
            by_q[fq].append(f)
    if quantization is None:
        quantization = prompt_quantization(quantization_choices(ggufs), input_fn=input_fn, output_fn=output_fn)
    else:
        quantization = normalize_quantization(quantization)
    matches = sorted(by_q[quantization], key=lambda x: (len(Path(x).name), x))
    if not matches:
        available = ", ".join(quantization_choices(ggufs)) or "none"
        raise FileNotFoundError(f"No {quantization} GGUF found. Available supported quantizations: {available}")
    return matches[0]


def select_tokenizer_sidecar(files: list[str], model_filename: str) -> str | None:
    candidates = [f for f in files if Path(f).name.lower().startswith("surface-vocab") and f.lower().endswith(".bin")]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    model_upper = Path(model_filename).name.upper()
    tags = re.findall(r"NDSRF[0-9A-Z]+", model_upper)
    for tag in tags:
        tagged = [f for f in candidates if tag in Path(f).name.upper()]
        if len(tagged) == 1:
            return tagged[0]
    return sorted(candidates, key=lambda x: (len(Path(x).name), x))[0]
