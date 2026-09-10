# Performance plan

The core is structured so architecture semantics are independent from compute backends.

## Implemented in alpha

- read-only memory mapping instead of eager full-weight copies
- direct F16, Q8_0 and Q4_0 GEMV paths
- F16C vector conversion when available
- AVX2/FMA Q8_0 and Q4_0 dot-product paths
- OpenMP row parallelism
- `-O3` native-ISA developer build; aggressive fast-math is opt-in only after parity
