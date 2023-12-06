# SpeechBrain_Quant

This repo provides different techniques for quantizing speechbrain models. Techniques the team might experiment with:

* Quantization post fine-tuning
* Quantization+fine-tuning at the same time (Quantization-Aware Training)
* Stochastic quantization
* Mixed-Precision Quantization (e.g., some layers are 8-bit, others 4-bit)
* Static vs Dynamic quantization
* Uniform vs Non-uniform quantization
* Symmetric vs Asymmetric Quantization
* Groupwise quantization
* Simulated and Integer-only Quantization
* Batch normalization folding for more efficient inference
* Adaptive Quantization
* Vector Quantization