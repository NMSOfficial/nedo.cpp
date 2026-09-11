from nedo.cli import _build_prompt, build_parser


def test_first_turn_prompt_uses_exact_sft_input_adapter():
    p = _build_prompt([], "Merhaba")
    assert p.startswith("Kullanıcı talimatı:\nKullanıcının şu mesajına doğal, kısa ve doğrudan Türkçe yanıt ver.")
    assert "\n\nEk bilgi:\nMerhaba\n\nAsistan cevabı:\n" in p
    assert "Son kullanıcı mesajı:" not in p
    assert p.count("Kullanıcı talimatı:") == 1
    assert p.count("Asistan cevabı:") == 1


def test_history_stays_inside_optional_input_and_is_bounded():
    history = [(f"u{i}", f"a{i}") for i in range(6)]
    p = _build_prompt(history, "Nasılsın?")
    assert "Önceki konuşma:" in p
    assert "Kullanıcı: u0" not in p
    assert "Kullanıcı: u1" not in p
    for i in range(2, 6):
        assert f"Kullanıcı: u{i}" in p
        assert f"Asistan: a{i}" in p
    assert "Son kullanıcı mesajı:\nNasılsın?" in p
    assert p.count("Kullanıcı talimatı:") == 1
    assert p.count("Asistan cevabı:") == 1


def test_chat_defaults_match_nedolm_repetition_controls():
    args = build_parser().parse_args(["chat", "Ethosoft/NedoLM-0.8B-SFT-6478-GGUF"])
    assert args.temperature == 0.0
    assert args.max_tokens == 128
    assert args.device == "auto"
    assert args.repetition_penalty == 1.15
    assert args.no_repeat_ngram_size == 4
