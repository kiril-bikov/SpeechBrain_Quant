## How to Perform Adaptive Rounding Optimization

1. Run `AdaRound_Activations_Wav2Vec2.ipynb`.
2. Make a directory named `activations` and store the `.pkl` files there.
3. Run `AdaRoundOptimize.ipynb`. This script learns rounding decisions (alpha) for weights.
4. Replace the updated `tensor_quant` and `tensor_quantizer` scripts in `TensorRT/tools/pytorch-quantization` and re-install.
5. Run `AdaRoundQuantizationWav2Vec2.ipynb`. This script quantizes the model and performs inference.