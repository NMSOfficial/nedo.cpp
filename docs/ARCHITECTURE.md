# NedoLM runtime notes

These notes capture the public contract used by nedo.cpp 0.1 alpha.

## Confirmed public shape

- architecture: `nedolm`
- 24 decoder blocks
- hidden width 1536
- 12 query heads, 4 KV heads, head dimension 128
- FFN width 5632
- context 4096
- sliding window 2048
- 18 of 24 blocks use TokenPrior MorphFFN
- routing channels are described as OTHER / ROOT / SUFFIX
- tied token embeddings

## Correctness gate

The exact public MorphFFN operation is not yet encoded in this alpha. Generation stays gated until reference parity is available.
