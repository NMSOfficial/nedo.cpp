# nedo.cpp

Independent C++20 runtime and Python SDK for NedoLM GGUF models.

> Alpha status: GGUF loading, mmap, F16/Q8_0/Q4_0 kernels, tokenizer surface layer, model-schema discovery, Python bindings and the `nedo chat` CLI are implemented. Text generation remains deliberately parity-gated until the exact NedoLM MorphFFN routing contract is implemented and validated.

## Install

```bash
pip install -U git+https://github.com/NMSOfficial/nedo.cpp.git
```

Host-native CPU build:

```bash
NEDO_NATIVE_BUILD=1 pip install -U git+https://github.com/NMSOfficial/nedo.cpp.git
```

`huggingface_hub` is installed automatically.

## Python

```python
import nedo.cpp as nedo

model = nedo.import_llm("Ethosoft/NedoLM-0.8B-SFT-GGUF")
print(model.summary())
```

If a Hugging Face repository exposes multiple supported GGUF quantizations, `nedo.cpp` lists them before downloading model bytes and asks you to select one numerically.

For non-interactive use:

```python
model = nedo.import_llm(
    "Ethosoft/NedoLM-0.8B-SFT-GGUF",
    quantization="Q8_0",
)
```

## CLI

```bash
nedo chat Ethosoft/NedoLM-0.8B-SFT-6478-GGUF
```

Use `/clear` to reset conversation history and `/exit` or `/quit` to leave.

## Build and test

```bash
cmake -S . -B build -DNEDO_BUILD_PYTHON=OFF -DNEDO_BUILD_TESTS=ON -DNEDO_NATIVE=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## License

Apache-2.0
