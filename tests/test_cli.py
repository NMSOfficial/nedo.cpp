from nedo.cli import _build_prompt, _generation_config, build_parser


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


def test_chat_defaults_reach_native_generation_config():
    args = build_parser().parse_args(["chat", "Ethosoft/NedoLM-0.8B-SFT-6478-GGUF"])
    assert args.max_tokens == 128
    assert args.device == "auto"
    assert args.temperature == 0.7
    assert args.top_p == 0.9
    assert args.top_k == 0
    assert args.repetition_penalty == 1.15
    assert args.no_repeat_ngram_size == 4

    cfg = _generation_config(args)
    assert cfg.max_new_tokens == 128
    assert abs(cfg.temperature - 0.7) < 1e-6
    assert abs(cfg.top_p - 0.9) < 1e-6
    assert cfg.top_k == 0
    assert abs(cfg.repetition_penalty - 1.15) < 1e-6
    assert cfg.no_repeat_ngram_size == 4
