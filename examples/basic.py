import nedo.cpp as nedo

model = nedo.import_llm("Ethosoft/NedoLM-0.8B-SFT-GGUF")
print(model.summary())
print(model.tokenize("Merhaba dünya"))
