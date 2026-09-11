from __future__ import annotations

import argparse
import sys
from . import cpp as nedo


_CHAT_STOPS = (
    "</düşünme>",
    "<düşünme>",
    "<|im_end|>",
    "<|end|>",
    "<|eot_id|>",
    "<|endoftext|>",
    "\nKullanıcı talimatı:",
    "Kullanıcı talimatı:",
    "\nAsistan cevabı:",
    "Asistan cevabı:",
)

_CHAT_INSTRUCTION = (
    "Kullanıcının şu mesajına doğal, kısa ve doğrudan Türkçe yanıt ver. "
    "Yalnızca cevabı yaz."
)
_MAX_HISTORY_TURNS = 4


def _build_prompt(history: list[tuple[str, str]], user_text: str) -> str:
    # NedoLM SFT is instruction+optional-input tuning, not role-based multi-turn
    # chat. The first turn deliberately keeps Ek bilgi equal to the user text;
    # this is the prompt shape validated against the real F16 checkpoint.
    if not history:
        context_text = user_text
    else:
        context: list[str] = ["Önceki konuşma:"]
        for user, assistant in history[-_MAX_HISTORY_TURNS:]:
            context.append(f"Kullanıcı: {user}")
            context.append(f"Asistan: {assistant.strip()}")
        context.extend(("", "Son kullanıcı mesajı:", user_text))
        context_text = "\n".join(context)
    return (
        f"Kullanıcı talimatı:\n{_CHAT_INSTRUCTION}\n\n"
        f"Ek bilgi:\n{context_text}\n\n"
        "Asistan cevabı:\n"
    )


def _generation_config(args: argparse.Namespace):
    cfg = nedo.GenerationConfig()
    cfg.max_new_tokens = args.max_tokens
    cfg.temperature = args.temperature
    cfg.top_p = args.top_p
    cfg.top_k = args.top_k
    cfg.repetition_penalty = args.repetition_penalty
    cfg.no_repeat_ngram_size = args.no_repeat_ngram_size
    cfg.seed = args.seed
    return cfg


def chat(args: argparse.Namespace) -> int:
    model = nedo.import_llm(
        args.model,
        filename=args.filename,
        quantization=args.quantization,
        revision=args.revision,
        device=args.device,
    )
    cfg = _generation_config(args)
    history: list[tuple[str, str]] = []
    if not args.quiet:
        print(model.summary(), file=sys.stderr)
        print("/exit veya /quit ile çıkın. /clear geçmişi temizler.", file=sys.stderr)
    while True:
        try:
            user_text = input("> ")
        except EOFError:
            break
        except KeyboardInterrupt:
            print(file=sys.stderr)
            break
        stripped = user_text.strip()
        if not stripped:
            continue
        if stripped.lower() in {"/exit", "/quit"}:
            break
        if stripped.lower() == "/clear":
            history.clear()
            if not args.quiet:
                print("Geçmiş temizlendi.", file=sys.stderr)
            continue
        prompt = _build_prompt(history, user_text)
        try:
            def emit(chunk: str) -> None:
                sys.stdout.write(chunk)
                sys.stdout.flush()
            text = model.generate_stream(prompt, cfg, on_text=emit, stop=_CHAT_STOPS)
        except RuntimeError as exc:
            print(f"nedo.cpp generation hatası: {exc}", file=sys.stderr)
            return 2
        if not text.endswith("\n"):
            sys.stdout.write("\n")
        sys.stdout.flush()
        history.append((user_text, text))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nedo", description="NedoLM GGUF runtime")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("chat", help="Interactive NedoLM SFT chat adapter")
    p.add_argument("model", help="Hugging Face repo id/URL or local GGUF")
    p.add_argument("--quantization", choices=["Q4_0", "Q8_0", "F16"])
    p.add_argument("--filename")
    p.add_argument("--revision")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Compute device; auto prefers CUDA when available")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--repetition-penalty", type=float, default=1.15, help="Penalize tokens already present in the active sequence")
    p.add_argument("--no-repeat-ngram-size", type=int, default=4, help="Block repeated token n-grams; 0 disables")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quiet", action="store_true", help="Hide model metadata/status; responses stay raw text")
    p.set_defaults(func=chat)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
