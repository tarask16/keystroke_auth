# Stage 7. Metric-learning embedding verification

## 1. Цель этапа

Цель Stage 7 — проверить, сможет ли metric-learning обучение сформировать embedding-пространство, более пригодное для distance-based biometric verification, чем encoder, обученный через softmax cross-entropy.

Stage 6 показал, что softmax-trained encoder обеспечивает высокую classification accuracy, но уступает softmax baseline v2 по FRR и balanced error при FAR около 1%.

## 2. Baseline для сравнения

Основной baseline:

| Подход | FAR | FRR | Balanced error | EER |
|---|---:|---:|---:|---:|
| Softmax baseline v2 | 1.0005% | 2.5245% | 1.7625% | 1.7100% |

Stage 6 embedding baseline:

| Подход | FAR | FRR | Balanced error | EER |
|---|---:|---:|---:|---:|
| Softmax-trained embedding, cosine + per-user | 0.9574% | 13.4559% | 7.2066% | 5.5152% |

## 3. Проверяемая гипотеза

Metric-learning objective должен уменьшить внутрипользовательские расстояния и увеличить межпользовательские расстояния, что должно снизить FRR при целевом FAR около 1%.

Гипотеза считается подтверждённой, если лучший Stage 7 вариант на test split покажет:

- FAR ≤ 1.1%;
- FRR < 5%;
- EER < 3%;
- balanced error < 3%;
- max user FAR не выше уровня baseline v2.

## 4. Подходы Stage 7

### 4.1 Siamese network + contrastive loss

Схема:

```text
sample_a -> shared_encoder -> embedding_a
sample_b -> shared_encoder -> embedding_b
distance(embedding_a, embedding_b)
contrastive loss
```

Проверить:

- Euclidean distance;
- cosine distance;
- разные margin;
- balanced genuine/impostor pairs.

### 4.2 Triplet-loss encoder

Схема:

```text
anchor   -> encoder -> embedding_anchor
positive -> encoder -> embedding_positive
negative -> encoder -> embedding_negative
triplet loss
```

Проверить:

- random negative mining;
- semi-hard negative mining;
- hard negative mining;
- разные margin.

### 4.3 Optional: classification + metric fine-tuning

Схема:

```text
softmax-trained encoder -> metric-learning fine-tuning
```

Идея: использовать Stage 6 encoder как initial weights, затем дообучить embedding layer под verification objective.

## 5. План задач

### 7.1 Pair / triplet dataset generation

Файлы:

```text
src/embedding_model/generate_metric_learning_pairs.py
src/embedding_model/generate_metric_learning_triplets.py
```

Артефакты:

```text
data/processed/embedding_model/metric_pairs_train.npz
data/processed/embedding_model/metric_pairs_validation.npz
data/processed/embedding_model/metric_triplets_train.npz
data/processed/embedding_model/metric_triplets_validation.npz
```

### 7.2 Siamese model

Файлы:

```text
src/embedding_model/siamese_model.py
src/embedding_model/train_siamese_embedding.py
```

Артефакты:

```text
models/embedding_model/siamese_model.keras
models/embedding_model/siamese_encoder.keras
reports/embedding_model/siamese_training_metrics.csv
```

### 7.3 Triplet model

Файлы:

```text
src/embedding_model/triplet_model.py
src/embedding_model/train_triplet_embedding.py
```

Артефакты:

```text
models/embedding_model/triplet_model.keras
models/embedding_model/triplet_encoder.keras
reports/embedding_model/triplet_training_metrics.csv
```

### 7.4 Generalized embedding evaluation pipeline

Обобщить stage 6 scripts под несколько encoder-ов:

```text
--encoder-name softmax
--encoder-name siamese
--encoder-name triplet
```

Каталоги результатов:

```text
reports/embedding_model/softmax_encoder/
reports/embedding_model/siamese_encoder/
reports/embedding_model/triplet_encoder/
```

### 7.5 Stage 7 comparison report

Файл:

```text
reports/embedding_model/stage7_metric_learning_report.md
```

Сравнить:

- softmax baseline v2;
- Stage 6 softmax-trained embedding;
- Siamese embedding;
- Triplet embedding;
- optional metric fine-tuned embedding.

## 6. Основные метрики

Для каждого варианта фиксировать:

- FAR;
- FRR;
- EER;
- ROC AUC;
- balanced error;
- false accepts;
- false rejects;
- max user FAR;
- max user FRR;
- enrollment sensitivity.

## 7. Ожидаемый результат

Минимально приемлемый результат Stage 7:

```text
FAR около 1%
FRR < 5%
EER < 3%
```

Сильный результат:

```text
FAR около 1%
FRR <= 3%
EER <= 2%
```

Научно значимый отрицательный результат:

```text
Даже metric-learning подход не превосходит softmax baseline v2 на CMU keystroke dynamics benchmark.
```

## 8. Риски

- несбалансированная генерация пар;
- слишком лёгкие impostor pairs;
- переобучение на train users;
- нестабильность hard negative mining;
- рост FRR при чрезмерно строгом margin;
- несопоставимость evaluation protocol с Stage 6.

## 9. Критерии закрытия Stage 7

Stage 7 считается завершённым, если:

- реализованы pair/triplet generators;
- обучены Siamese и Triplet encoder-ы;
- выполнена evaluation по тому же protocol, что Stage 6;
- сформирован `stage7_metric_learning_report.md`;
- есть pytest/quality checks;
- сделано сравнение с softmax baseline v2 и Stage 6 embedding baseline.
