# SpeechBrain_Quant

This repo provides different techniques for quantizing SpeechBrain CRDNN and Wav2Vec2 on LibriSpeech. Quantization techniques with evaluation provided:

* Post Training Quantization
* Mixed-precision Quantization(some layers are 8-bit, others 4-bit)
* Quantization-aware Training
* Stochastic Quantization
* AdaRound
* Batch Normalization Folding after Post Training Quantization