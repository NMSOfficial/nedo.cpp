from nedo.cli import _build_prompt


def test_first_turn_prompt():
    assert _build_prompt([], "Merhaba") == "Kullanıcı talimatı:\nMerhaba\n\nAsistan cevabı:\n"


def test_history_prompt():
    p = _build_prompt([("Merhaba", "Selam")], "Nasılsın?")
    assert p == (
        "Kullanıcı talimatı:\nMerhaba\n\nAsistan cevabı:\nSelam\n\n"
        "Kullanıcı talimatı:\nNasılsın?\n\nAsistan cevabı:\n"
    )
