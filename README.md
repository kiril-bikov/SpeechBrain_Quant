# SpeechBrain_Quant

This repository provides code for quantizing SpeechBrain CRDNN and Wav2Vec2 on LibriSpeech. Quantization techniques with evaluation provided:

* Post Training Quantization
* Mixed-precision Quantization(some layers are 8-bit, others 4-bit)
* Quantization-aware Training
* Stochastic Quantization
* AdaRound
* Batch Normalization Folding after Post Training Quantization

The repository is split in two folders  - `Main Experiments` and `Other Experiments`.

The `Main Experiments` are structured, performed on GPU over the LibriSpeech dataset using the CRDNN and Wav2Vec2 models.

The `Other Experiments` are unstructured, performed on GPU and CPU over a variety of datasets using the CRDNN and Wav2Vec2 models.
