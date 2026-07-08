# MLP Architecture Comparison

## Summary

- best by EER: `mlp_256_128`;
- best EER: `0.016480`;
- best by test accuracy: `mlp_128_64_batchnorm`;
- best test accuracy: `0.917892`.

## Results

| architecture | hidden_layers | dropout_rate | batch_norm | params | epochs_run | best_validation_accuracy | final_train_accuracy | final_validation_accuracy | test_loss | test_accuracy | eer | eer_threshold | target_far | threshold_at_target_far | actual_far | actual_frr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlp_64 | 64 | 0.200000 | False | 5363 | 20 | 0.856618 | 0.830729 | 0.856618 | 0.480260 | 0.870343 | 0.026007 | 0.034142 | 0.010000 | 0.107868 | 0.010005 | 0.055392 |
| mlp_128_64 | 128-64 | 0.200000 | False | 15667 | 20 | 0.886336 | 0.844669 | 0.886336 | 0.362027 | 0.894608 | 0.019684 | 0.029313 | 0.010000 | 0.073476 | 0.010005 | 0.033824 |
| mlp_256_128 | 256-128 | 0.300000 | False | 47667 | 20 | 0.907782 | 0.878523 | 0.907782 | 0.309213 | 0.909314 | 0.016480 | 0.025448 | 0.010000 | 0.053864 | 0.010005 | 0.025245 |
| mlp_128_64_dropout | 128-64 | 0.350000 | False | 15667 | 20 | 0.871630 | 0.765931 | 0.871630 | 0.440348 | 0.880147 | 0.022787 | 0.039402 | 0.010000 | 0.099274 | 0.010005 | 0.044853 |
| mlp_128_64_batchnorm | 128-64 | 0.200000 | True | 16243 | 20 | 0.909007 | 0.859988 | 0.909007 | 0.303182 | 0.917892 | 0.017142 | 0.027535 | 0.010000 | 0.058608 | 0.010005 | 0.025245 |
