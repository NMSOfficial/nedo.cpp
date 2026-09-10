# Release checklist

A release that enables `Model.generate()` must satisfy all of the following:

- [ ] exact MorphFFN equation and canonical tensor mapping documented
- [ ] NDSRF004 full tokenizer parity vectors pass
- [ ] F16 final-logit parity against the reference runtime
- [ ] greedy generation parity on fixed Turkish prompts
- [ ] Q8_0/Q4_0 numerical regression suite passes
- [ ] native and portable CPU builds pass CI on Linux/macOS/Windows
