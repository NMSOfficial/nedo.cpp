from nedo.cli import _build_prompt, build_parser


def test_first_turn_prompt():
    assert _build_prompt([], "Merhaba") == "Kullanıcı talimatı:\nMerhaba\n\nAsistan cevabı:\n"


def test_history_uses_optional_input_field():
    p = _build_prompt([("Merhaba", "Selam")], "Nasılsın?")
    assert p == (
        "Kullanıcı talimatı:\nNasılsın?\n\n"
        "Ek bilgi:\nÖnceki konuşma:\n"
        "Kullanıcı: Merhaba\nAsistan: Selam\n\n"
        "Asistan cevabı:\n"
    )
    assert p.count("Kullanıcı talimatı:") == 1
    assert p.count("Asistan cevabı:") == 1


def test_chat_defaults_to_deterministic_greedy():
    args = build_parser().parse_args(["chat", "Ethosoft/NedoLM-0.8B-SFT-6478-GGUF"])
    assert args.temperature == 0.0
    assert args.device == "auto"
