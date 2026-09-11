from nedo.cpp import _StopTextBuffer


def test_stop_marker_never_leaks_across_chunks():
    out: list[str] = []
    buf = _StopTextBuffer(("Kullanıcı talimatı:",), out.append)
    assert buf.feed("Merhaba\n\nKullanıcı tali") is False
    assert buf.feed("matı:\nsonraki tur") is True
    assert buf.stopped is True
    assert "".join(out) == "Merhaba\n\n"


def test_non_stop_text_flushes_losslessly():
    out: list[str] = []
    buf = _StopTextBuffer(("</düşünme>",), out.append)
    assert buf.feed("Normal bir cevap") is False
    buf.finish()
    assert "".join(out) == "Normal bir cevap"
