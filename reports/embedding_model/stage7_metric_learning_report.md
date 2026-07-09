# Stage 7. Metric learning для embedding-based verification

## 1. Назначение этапа

Цель Stage 7 — проверить, может ли metric-learning подход улучшить embedding-based verification по сравнению с Stage 6 и baseline v2.

Проверялись два семейства моделей:

- **Siamese encoder** с contrastive loss;
- **Triplet encoder** с triplet loss.

Основная схема проверки:

```text
feature vector → encoder → embedding → distance to user_template → threshold_user → ACCEPT / REJECT
```

Ключевой критерий — качество аутентификации при целевом уровне FAR около 1%.

---

## 2. Контрольные точки сравнения

| Подход | FAR | FRR | Balanced error | EER | Комментарий |
|---|---:|---:|---:|---:|---|
| Softmax baseline v2 | 1.0005% | 2.5245% | 1.7625% | 1.7100% | Лучший контрольный baseline |
| Stage 6 embedding best | 0.9574% | 13.4559% | 7.2066% | 5.5152% | Softmax-trained encoder + templates |

Baseline v2 остаётся основной контрольной точкой, так как он использует softmax score заявленного пользователя напрямую и показывает существенно меньший FRR.

---

## 3. Целевые критерии Stage 7

Stage 7 считался успешным, если metric-learning encoder достигает следующих критериев:

| Метрика | Целевое значение |
|---|---:|
| FAR | ≤ 1.1% |
| FRR | < 5% |
| EER | < 3% |
| Balanced error | < 3% |

Дополнительное требование: результат должен быть лучше Stage 6 embedding baseline и приближаться к softmax baseline v2.

---

## 4. Реализованные подэтапы

### 4.1. Task 7.1 — Pair / triplet dataset generation

Реализована генерация обучающих наборов:

- balanced genuine/impostor pairs;
- random triplets;
- train/validation split без использования test split;
- отдельный scaler, обученный только на train split.

Сформированные наборы:

| Набор | Количество |
|---|---:|
| Train pairs | 102000 |
| Validation pairs | 25500 |
| Train triplets | 51000 |
| Validation triplets | 12750 |
| Feature dimension | 31 |

### 4.2. Task 7.2 — Siamese encoder

Реализована Siamese-модель:

- shared encoder;
- Euclidean distance output;
- contrastive loss;
- сохранение полного model и encoder;
- CSV training metrics.

Финальный одиночный запуск `margin=1.0`:

| Метрика | Значение |
|---|---:|
| Train loss | 0.07745 |
| Validation loss | 0.07607 |
| Train contrastive accuracy | 92.24% |
| Validation contrastive accuracy | 91.43% |

### 4.3. Task 7.3 — Triplet encoder

Реализована Triplet-модель:

- shared encoder;
- triplet loss;
- triplet margin accuracy;
- сохранение полного model и encoder;
- CSV training metrics.

Финальный одиночный запуск `margin=0.2`:

| Метрика | Значение |
|---|---:|
| Train loss | 0.00844 |
| Validation loss | 0.01014 |
| Train triplet margin accuracy | 93.41% |
| Validation triplet margin accuracy | 93.87% |

### 4.4. Task 7.4 — Unified encoder evaluation

Реализован общий evaluation pipeline для encoder-ов:

- `softmax`;
- `siamese`;
- `triplet`.

Поддержаны расстояния:

- cosine;
- euclidean;
- manhattan.

Поддержаны политики порогов:

- global;
- per-user;
- guarded.

Для каждого encoder-а формировались:

- `embedding_distance_diagnostics.csv`;
- `embedding_threshold_policy.csv`;
- `user_templates.json`;
- `user_thresholds.json`.

### 4.5. Task 7.5 — Margin tuning

Выполнен margin sweep для Siamese и Triplet.

Siamese margins:

```text
0.2, 0.5, 1.0, 1.5
```

Triplet margins:

```text
0.1, 0.2, 0.5, 1.0
```

### 4.6. Task 7.6 — Hard / semi-hard negative mining

Реализован эксперимент hard/semi-hard negative mining:

- mining через seed encoder;
- стратегии `hard`, `semi_hard`, `mixed`;
- отдельные hard-mined triplet-наборы;
- manifest CSV с диагностикой mining-а;
- полный training/evaluation pipeline.

---

## 5. Итоговые результаты Stage 7

### 5.1. Лучшие результаты по семействам моделей

| Подход | Лучший кандидат | Distance | Policy | FAR | FRR | Balanced error | EER | ROC AUC |
|---|---|---|---|---:|---:|---:|---:|---:|
| Softmax encoder re-eval | softmax encoder | cosine | per_user | 1.1667% | 17.7941% | 9.4804% | 6.0532% | 98.4474% |
| Triplet random sweep | triplet_m0p1_l2 | cosine | per_user | 1.2152% | 17.1078% | 9.1615% | 5.9836% | 98.6392% |
| Siamese sweep | siamese_m0p5_l2 | cosine | per_user | 1.1113% | 33.9706% | 17.5409% | 8.1108% | 97.2871% |
| Hard negative mining | triplet_hard_m0p2_top32_cosine_l2 | cosine | per_user | 1.1078% | 52.8431% | 26.9755% | 17.2110% | 90.6997% |
| Semi-hard negative mining | triplet_semihard_m0p2_top32_cosine_l2 | cosine | per_user | 1.0078% | 58.7500% | 29.8789% | 20.5880% | 87.7998% |

### 5.2. Лучший общий результат Stage 7

Лучший результат среди Stage 7 metric-learning экспериментов:

| Candidate | Objective | Margin | Distance | Policy | FAR | FRR | Balanced error | EER |
|---|---|---:|---|---|---:|---:|---:|---:|
| triplet_m0p1_l2 | triplet | 0.1 | cosine | per_user | 1.2152% | 17.1078% | 9.1615% | 5.9836% |

Этот результат лучше одиночного Triplet `margin=0.2`, но хуже Stage 6 embedding baseline и существенно хуже softmax baseline v2.

---

## 6. Сравнение с baseline v2 и Stage 6

| Подход | FAR | FRR | Balanced error | EER | Итог |
|---|---:|---:|---:|---:|---|
| Softmax baseline v2 | 1.0005% | 2.5245% | 1.7625% | 1.7100% | Лучший результат |
| Stage 6 embedding best | 0.9574% | 13.4559% | 7.2066% | 5.5152% | Лучше Stage 7 metric learning |
| Stage 7 best metric learning | 1.2152% | 17.1078% | 9.1615% | 5.9836% | Цели не достигнуты |
| Stage 7 hard negative mining best | 1.1078% | 52.8431% | 26.9755% | 17.2110% | Существенное ухудшение |

---

## 7. Основные наблюдения

### 7.1. Metric learning обучается, но не даёт нужного verification trade-off

Siamese и Triplet модели демонстрируют нормальную динамику training loss и validation loss. Однако хорошее значение training objective не переносится напрямую на template-based verification.

Главный симптом:

```text
FAR удерживается около 1%, но FRR остаётся слишком высоким.
```

Это означает, что модели становятся достаточно строгими к impostor-попыткам, но слишком часто отклоняют genuine-попытки.

### 7.2. Triplet лучше Siamese

Лучший Triplet-кандидат:

```text
FRR = 17.1078%, EER = 5.9836%
```

Лучший Siamese-кандидат:

```text
FRR = 33.9706%, EER = 8.1108%
```

Следовательно, в данной постановке triplet loss оказался более перспективным, чем contrastive loss.

### 7.3. Увеличение margin ухудшает usability

Для Triplet margin sweep наблюдалась закономерность:

```text
margin ↑ → FAR немного ↓, но FRR резко ↑
```

Наиболее мягкий вариант `margin=0.1` дал лучший balanced error.

### 7.4. Hard negative mining ухудшил результат

Hard/semi-hard mining в текущей реализации резко увеличил FRR:

```text
Random Triplet best FRR: 17.1078%
Hard mining best FRR:   52.8431%
Semi-hard best FRR:     58.7500%
```

Практический вывод: hard mining не следует использовать как полную замену random triplets. Если развивать это направление, нужен curriculum или mixed mining с небольшой долей hard examples.

### 7.5. Softmax-score authentication остаётся лучшей политикой

Baseline v2 использует softmax score заявленного пользователя напрямую и показывает существенно лучший verification trade-off:

```text
FAR = 1.0005%
FRR = 2.5245%
EER = 1.7100%
```

Это значительно лучше всех template-based embedding вариантов.

---

## 8. Проверка целевых критериев

| Критерий | Цель | Лучший Stage 7 результат | Статус |
|---|---:|---:|---|
| FAR | ≤ 1.1% | 1.2152% | FAIL |
| FRR | < 5% | 17.1078% | FAIL |
| EER | < 3% | 5.9836% | FAIL |
| Balanced error | < 3% | 9.1615% | FAIL |

Stage 7 не достиг целевых критериев.

---

## 9. Итоговый вывод Stage 7

Stage 7 завершён как отрицательный, но информативный эксперимент.

Ключевой вывод:

```text
В текущей постановке metric-learning подходы Siamese и Triplet не улучшают embedding-based verification для CMU fixed-text keystroke dynamics benchmark и не превосходят softmax baseline v2.
```

Лучшим решением проекта на текущем этапе остаётся baseline v2:

```text
MLP softmax classifier + authentication by claimed-user softmax score.
```

Stage 6 и Stage 7 показывают, что переход к template-based embedding verification требует более сложной постановки, чем простое обучение encoder-а через contrastive/triplet loss.

---

## 10. Рекомендации для дальнейшей работы

### 10.1. Для инженерной версии системы

Использовать baseline v2 как основную рабочую политику:

```text
feature vector → scaler → MLP → softmax score of claimed user → threshold → ACCEPT / REJECT
```

Причина: минимальный FRR при FAR около 1%.

### 10.2. Для научной статьи

Stage 7 следует включать как отрицательный эксперимент:

- softmax baseline v2 существенно превосходит metric-learning варианты;
- Stage 6 embedding показывает ухудшение относительно direct softmax-score authentication;
- Stage 7 margin tuning не решает проблему высокого FRR;
- hard negative mining ухудшает verification metrics;
- результат полезен для обоснования выбора softmax-score authentication как практического baseline.

### 10.3. Если продолжать embedding-направление

Потенциальные направления:

1. fine-tuning Stage 6 encoder-а, а не обучение metric-learning encoder-а с нуля;
2. center loss / supervised contrastive loss;
3. ArcFace / CosFace-подобные функции потерь;
4. прототипные сети с template-aware loss;
5. calibration по пользователям;
6. score fusion: softmax score + embedding distance;
7. mixed mining вместо hard-only mining.

---

## 11. Статус Stage 7

```text
[OK] 7.1 Pair / triplet dataset generation
[OK] 7.2 Siamese model
[OK] 7.3 Triplet model
[OK] 7.4 Unified evaluation pipeline
[OK] 7.5 Margin tuning experiment
[OK] 7.6 Hard / semi-hard negative mining experiment
[OK] 7.7 Final Stage 7 comparison report
[FAIL] Целевые метрики Stage 7 не достигнуты
```

Stage 7 закрыт.
