## WER and inference time results of the tested qunatization methods on CRDNN and Wav2Vec2

### CRDNN Post Training Quantization

| CRDNN               | WER   | Time(sec) |
|---------------------|-------|-----------|
| Base                | 3.16  | 1.1       |
| Quantized 8 bit MAX | 3.58  | 0.6       |
| Quantized 8 bit MSE | 3.73  | -         |


### Wav2Vec2 Post Training Quantization

| Wav2Vec2    | WER   | Time(sec) |
|-------------|-------|-----------|
| Base        | 2.29  | 0.47      |
| Quantized   | 2.38  | 0.02      |

### CRDNN Mixed-Precision Quantization

| CRDNN                    | WER  | Time(sec) |
|--------------------------|------|-----------|
| Base                     | 3.16 | 0.41      |
| All 8 bit                | 3.5  | 0.172     |
| All 4 bit                | 4.4  | 0.17      |
| 4 bits CNN, 8 bits DNN   | 3.7  | 0.171     |
| 8 bits CNN, 4 bits DNN   | 4.35 | 0.171     |


### Wav2Vec2 Mixed-Precision Quantization

| Wav2Vec2                 | WER  | Time(sec) |
|--------------------------|------|-----------|
| Base                     | 2.29 | 0.47      |
| All 8 bit                | 2.38 | 0.021     |
| All 4 bit                | 4.51 | 0.02      |
| 4 bits Input, 8 bit Weight | 4.57 | 0.02    |
| 8 bits Input, 4 bit Weight | 2.42 | 0.02    |

### CRDNN Quantization-Aware Training

| Fine-tuned CRDNN | WER   | Time(sec) |
|------------------|-------|-----------|
| Base             | 3.23  | 0.488     |
| Quantized        | 3.27  | 0.174     |


### Wav2Vec2 Quantization-Aware Training

| Fine-tuned Wav2Vec2 | WER   | Time(sec) |
|---------------------|-------|-----------|
| Base                | 2.37  | 0.46      |
| Quantized           | 3.64  | 0.02      |


### CRDNN Stochastic QAT

| Fine-tuned CRDNN      | WER   | Time(sec) |
|-----------------------|-------|-----------|
| Base                  | 3.23  | 0.488     |
| Quantized             | 3.27  | 0.174     |
| Quantized Stochastic  | 3.24  | 0.164     |

### Wav2Vec2 Stochastic QAT

| Fine-tuned Wav2Vec2   | WER   | Time(sec) |
|-----------------------|-------|-----------|
| Base                  | 2.37  | 0.46      |
| Quantized             | 3.64  | 0.02      |
| Quantized Stochastic  | 2.69  | 0.02      |


### CRDNN AdaRound Quantization

| CRDNN               | WER   | Time(sec) |
|---------------------|-------|-----------|
| Base                | 3.16  | 0.41      |
| Quantized           | 3.5   | 0.172     |
| AdaRound Quantized  | 3.4   | 0.172     |

### Wav2Vec2 AdaRound Quantization

| Wav2Vec2            | WER   | Time(sec) |
|---------------------|-------|-----------|
| Base                | 2.29  | 0.47      |
| Quantized           | 2.38  | 0.02      |
| AdaRound Quantized  | 2.29  | 0.02      |


### BN Folding after PTQ on CRDNN

| Model                     | WER  | Time(sec) |
|---------------------------|------|-----------|
| Base                      | 3.16 | 0.41      |
| Quantized                 | 3.5  | 0.172     |
| Quantized with BN Folding | 3.8  | 0.04      |
