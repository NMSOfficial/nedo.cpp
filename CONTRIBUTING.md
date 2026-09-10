# Contributing

Correctness before throughput. Any MorphFFN implementation must include: (1) documented tensor shape/name mapping, (2) tokenizer ID parity vectors, (3) logits parity against the reference runtime on fixed prompts, and (4) benchmark results before/after an optimization. Avoid architecture semantics inside SIMD/GPU kernels.
