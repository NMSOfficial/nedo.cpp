from nedo.cli import _build_prompt, build_parser


def test_first_turn_prompt():
    assert _build_prompt([], "Merhaba") == "Kullanıcı talimatı:\nMerhaba\n\nAsistan cevabı:\n"


def test_history_prompt():
    p = _build_prompt([("Merhaba", "Selam")], "Nasılsın?")
    assert p == (
        "Kullanıcı talimatı:\nMerhaba\n\nAsistan cevabı:\nSelam\n\n"
        "Kullanıcı talimatı:\nNasılsın?\n\nAsistan cevabı:\n"
    )


def test_chat_defaults_to_deterministic_greedy():
    args = build_parser().parse_args(["chat", "Ethosoft/NedoLM-0.8B-SFT-6478-GGUF"])
    assert args.temperature == 0.0
    assert args.device == "auto"
