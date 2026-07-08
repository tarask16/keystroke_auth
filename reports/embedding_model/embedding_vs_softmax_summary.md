# Сравнение embedding-based verification с softmax baseline v2

## Лучший embedding-вариант

- Вариант: `mean_template_full_train_256_samples`
- Метрика расстояния: `cosine`
- Threshold policy: `per_user`
- Enrollment samples: `256`
- FAR: 0.9574%
- FRR: 13.4559%
- Balanced error: 7.2066%
- False accepts: 1953
- False rejects: 549

## Softmax baseline v2

- Вариант: `baseline_v2_mlp_128_64_batchnorm`
- FAR: 1.0005%
- FRR: 2.5245%
- Balanced error: 1.7625%
- EER: 1.7100%
- False accepts: 2041
- False rejects: 103

## Вывод

Итог: **embedding хуже baseline v2**.

Разница относительно softmax baseline v2:

- FAR delta: -0.0431 п.п.
- FRR delta: +10.9314 п.п.
- False accepts delta: -88
- False rejects delta: +446

Методологическая интерпретация:

Текущий encoder обучался через cross-entropy классификацию, поэтому его embedding-пространство не оптимизировано напрямую для distance-based verification. Это объясняет рост FRR при целевом FAR около 1%.
