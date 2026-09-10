from nedo._hub import (
    parse_hf_source,
    prompt_quantization,
    quantization_choices,
    select_gguf,
    select_tokenizer_sidecar,
)


def test_repo_id():
    src = parse_hf_source("Ethosoft/NedoLM-0.8B-SFT-GGUF")
    assert src.repo_id == "Ethosoft/NedoLM-0.8B-SFT-GGUF"
    assert src.filename is None


def test_full_url():
    src = parse_hf_source("https://huggingface.co/Ethosoft/NedoLM-0.8B-SFT-GGUF")
    assert src.repo_id == "Ethosoft/NedoLM-0.8B-SFT-GGUF"


def test_blob_url():
    src = parse_hf_source("https://huggingface.co/Ethosoft/NedoLM-0.8B-SFT-GGUF/blob/dev/model-Q8_0.gguf")
    assert src.repo_id == "Ethosoft/NedoLM-0.8B-SFT-GGUF"
    assert src.revision == "dev"
    assert src.filename == "model-Q8_0.gguf"


def test_quantization_choices_are_display_order_not_auto_selection():
    files = ["model-F16.gguf", "model-Q8_0.gguf", "model-Q4_0.gguf", "model-Q4_K_M.gguf"]
    assert quantization_choices(files) == ["Q4_0", "Q8_0", "F16"]


def test_interactive_selection_asks_before_choice():
    files = ["model-F16.gguf", "model-Q8_0.gguf", "model-Q4_0.gguf"]
    output: list[str] = []
    prompts: list[str] = []
    def fake_input(prompt: str) -> str:
        prompts.append(prompt)
        return "2"
    selected = select_gguf(files, input_fn=fake_input, output_fn=output.append)
    assert selected == "model-Q8_0.gguf"
    assert output[:4] == ["Bulunan quantization seçenekleri:", "1) Q4_0", "2) Q8_0", "3) F16"]
    assert prompts == ["Hangisini seçersiniz? [1-3]: "]


def test_interactive_selection_retries_invalid_number():
    answers = iter(["9", "x", "1"])
    output: list[str] = []
    selected = prompt_quantization(["Q4_0", "Q8_0"], input_fn=lambda _: next(answers), output_fn=output.append)
    assert selected == "Q4_0"
    assert output.count("Geçersiz seçim. 1 ile 2 arasında bir sayı girin.") == 2


def test_explicit_quantization_skips_prompt():
    files = ["model-F16.gguf", "model-Q8_0.gguf", "model-Q4_0.gguf"]
    def should_not_be_called(_: str) -> str:
        raise AssertionError("input() should not be called for explicit quantization")
    assert select_gguf(files, quantization="Q8_0", input_fn=should_not_be_called) == "model-Q8_0.gguf"


def test_filename_skips_prompt():
    files = ["model-Q8_0.gguf", "model-Q4_0.gguf"]
    assert select_gguf(files, filename="model-Q4_0.gguf") == "model-Q4_0.gguf"


def test_sidecar_matches_ndsrf_tag():
    files = ["surface-vocab-NDSRF003.bin", "surface-vocab-NDSRF004.bin", "NedoLM-F16-NDSRF004.gguf"]
    assert select_tokenizer_sidecar(files, "NedoLM-F16-NDSRF004.gguf") == "surface-vocab-NDSRF004.bin"
