# nedo.cpp

Independent C++20 runtime and Python SDK for NedoLM GGUF models.

Alpha status: native NedoLM generation, MorphFFN routing, canonical NDSRF004 tokenization, streaming chat, F16/Q8_0/Q4_0 CPU kernels and an optional CUDA matvec backend are implemented. The runtime is independent and is not a llama.cpp fork.

## Install

Use a virtual environment on distributions that protect the system Python (for example Arch Linux):

```bash
python -m venv ~/.venvs/nedo
~/.venvs/nedo/bin/pip install -U git+https://github.com/NMSOfficial/nedo.cpp.git
```

Host-native CPU build:

```bash
NEDO_NATIVE_BUILD=1 ~/.venvs/nedo/bin/pip install -U git+https://github.com/NMSOfficial/nedo.cpp.git
```

`huggingface_hub` and the canonical Ethosoft NedoTokenizer dependency are installed automatically.

## CUDA (NVIDIA, alpha)

When `nvcc` is available, the source build automatically compiles the CUDA backend. On Arch Linux the toolkit is normally installed under `/opt/cuda`:

```bash
sudo pacman -S cuda
CUDACXX=/opt/cuda/bin/nvcc NEDO_CUDA_BUILD=1 \
  ~/.venvs/nedo/bin/pip install -U --force-reinstall \
  git+https://github.com/NMSOfficial/nedo.cpp.git
```

Check the backend:

```bash
~/.venvs/nedo/bin/python -c 'import nedo.cpp as n; print(n.cuda_available(), n.cuda_device_name())'
```

`device="auto"` prefers CUDA when the extension was built with CUDA and an NVIDIA device is available. `device="cpu"` forces the portable CPU path. In the current alpha CUDA backend, GGUF F16/Q8_0/Q4_0 matrix-vector operations are offloaded and model weights are cached in VRAM; lightweight control, normalization and attention bookkeeping remain host-side.

## Python

```python
import nedo.cpp as nedo

model = nedo.import_llm(
    "Ethosoft/NedoLM-0.8B-SFT-6478-GGUF",
    device="auto",
)
print(model.summary())
```

Force CUDA:

```python
model = nedo.import_llm(
    "Ethosoft/NedoLM-0.8B-SFT-6478-GGUF",
    quantization="F16",
    device="cuda",
)
```

If a Hugging Face repository exposes multiple supported GGUF quantizations, `nedo.cpp` lists them before downloading model bytes and asks you to select one numerically. Explicit `quantization=` bypasses the prompt.

## CLI

```bash
nedo chat Ethosoft/NedoLM-0.8B-SFT-6478-GGUF --device auto
```

Force NVIDIA CUDA:

```bash
nedo chat Ethosoft/NedoLM-0.8B-SFT-6478-GGUF --device cuda
```

Use `/clear` to reset conversation history and `/exit` or `/quit` to leave. Generated UTF-8 text is streamed as tokens arrive.

## Build and test

CPU:

```bash
cmake -S . -B build -DNEDO_BUILD_PYTHON=OFF -DNEDO_BUILD_TESTS=ON -DNEDO_NATIVE=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Require CUDA (configuration fails instead of silently falling back if `nvcc` is missing):

```bash
NEDO_CUDA_BUILD=1 CUDACXX=/opt/cuda/bin/nvcc \
  cmake -S . -B build-cuda -DNEDO_BUILD_PYTHON=OFF -DNEDO_BUILD_TESTS=ON
cmake --build build-cuda -j
```

## License

Apache-2.0
