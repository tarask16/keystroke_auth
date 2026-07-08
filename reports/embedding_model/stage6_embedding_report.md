# Stage 6. Embedding-based verification

## 1. Назначение этапа

Цель этапа — проверить, можно ли перейти от softmax-based аутентификации к embedding-based verification, где решение принимается по расстоянию между текущим embedding-вектором и шаблоном заявленного пользователя.

Проверенная схема:

```text
sample -> encoder -> embedding vector -> distance to user_template -> threshold_user -> ACCEPT/REJECT
```

Базовая embedding-модель:

```text
Input(31 timing features)
Dense(128, ReLU)
BatchNormalization
Dropout(0.2)
Dense(32, ReLU, name='embedding')
BatchNormalization
Dense(51, Softmax)
```

Test accuracy embedding-классификатора: 92.0833%.

## 2. Качество embedding-классификатора

| Split      |   Samples | Accuracy   | Macro F1   | Weighted F1   |
|:-----------|----------:|:-----------|:-----------|:--------------|
| train      |     13056 | 97.7252%   | 97.7245%   | 97.7245%      |
| validation |      3264 | 91.5135%   | 91.4163%   | 91.4163%      |
| test       |      4080 | 92.0833%   | 92.0572%   | 92.0572%      |

Классификационное качество encoder-а достаточно высокое, но эта метрика не эквивалентна качеству biometric verification. Для аутентификации основными являются FAR, FRR и EER.

## 3. Диагностика embedding-расстояний

| Split      | Distance   |   ROC AUC | EER     |   Genuine mean |   Impostor mean |    Margin |
|:-----------|:-----------|----------:|:--------|---------------:|----------------:|----------:|
| validation | euclidean  |  0.980159 | 6.7953% |       6.05884  |       12.9662   |  6.90737  |
| validation | cosine     |  0.98664  | 5.8578% |       0.106372 |        0.471537 |  0.365165 |
| validation | manhattan  |  0.984718 | 6.0049% |      25.7254   |       57.5911   | 31.8657   |
| test       | euclidean  |  0.97852  | 6.8985% |       6.13423  |       13.0347   |  6.90049  |
| test       | cosine     |  0.986478 | 5.5152% |       0.107252 |        0.472495 |  0.365243 |
| test       | manhattan  |  0.983109 | 6.1304% |      25.9491   |       57.819    | 31.8699   |

Лучшая метрика по test EER: `cosine` с EER 5.5152%.

## 4. Threshold policy

Пороги подбирались только по validation split. Test split использовался только для финальной оценки.

| Distance   | Policy   | FAR     | FRR      | Balanced error   |   False accepts |   False rejects | Max user FAR   | Max user FRR   |
|:-----------|:---------|:--------|:---------|:-----------------|----------------:|----------------:|:---------------|:---------------|
| euclidean  | global   | 0.9338% | 21.3971% | 11.1654%         |            1905 |             873 | 5.0750%        | 67.5000%       |
| euclidean  | per_user | 0.9431% | 16.8137% | 8.8784%          |            1924 |             686 | 1.4000%        | 50.0000%       |
| euclidean  | guarded  | 0.8877% | 21.4216% | 11.1547%         |            1811 |             874 | 5.0750%        | 67.5000%       |
| cosine     | global   | 0.9387% | 15.2451% | 8.0919%          |            1915 |             622 | 3.2500%        | 52.5000%       |
| cosine     | per_user | 0.9574% | 13.4559% | 7.2066%          |            1953 |             549 | 1.4750%        | 52.5000%       |
| cosine     | guarded  | 0.8833% | 15.2941% | 8.0887%          |            1802 |             624 | 3.2500%        | 52.5000%       |
| manhattan  | global   | 0.8985% | 21.6667% | 11.2826%         |            1833 |             884 | 3.4500%        | 70.0000%       |
| manhattan  | per_user | 0.9495% | 16.5441% | 8.7468%          |            1937 |             675 | 1.6000%        | 51.2500%       |
| manhattan  | guarded  | 0.8279% | 21.7157% | 11.2718%         |            1689 |             886 | 3.2750%        | 70.0000%       |

Лучшая test-policy: `cosine + per_user`. FAR = 0.9574%, FRR = 13.4559%.

## 5. Эксперимент с количеством enrollment samples

Ниже показан основной вариант `cosine + per_user`.

|   N | FAR     | FRR      | Balanced error   |   False accepts |   False rejects | EER     |   ROC AUC |
|----:|:--------|:---------|:-----------------|----------------:|----------------:|:--------|----------:|
|   5 | 0.9750% | 20.2696% | 10.6223%         |            1989 |             827 | 6.8113% |  0.981319 |
|  10 | 0.9711% | 15.9069% | 8.4390%          |            1981 |             649 | 5.8025% |  0.985262 |
|  20 | 0.9716% | 14.6078% | 7.7897%          |            1982 |             596 | 5.7895% |  0.985412 |
|  30 | 0.9461% | 14.1422% | 7.5441%          |            1930 |             577 | 5.7363% |  0.986051 |
|  50 | 0.9725% | 13.8480% | 7.4103%          |            1984 |             565 | 5.5346% |  0.985761 |

Лучший enrollment-вариант по test balanced error: N = `50`, distance = `cosine`, policy = `per_user`.

Увеличение числа enrollment samples снижает FRR, но даже при N=50 FRR остаётся существенно выше softmax baseline v2.

## 6. Сравнение с softmax baseline v2

| Approach            | Variant                              | Metric        | Policy   | Enrollment   | FAR     | FRR      | Balanced error   | EER     |   False accepts |   False rejects |
|:--------------------|:-------------------------------------|:--------------|:---------|:-------------|:--------|:---------|:-----------------|:--------|----------------:|----------------:|
| Softmax baseline v2 | baseline_v2_mlp_128_64_batchnorm     | softmax_score | global   | -            | 1.0005% | 2.5245%  | 1.7625%          | 1.7100% |            2041 |             103 |
| Best embedding      | mean_template_full_train_256_samples | cosine        | per_user | 256.0        | 0.9574% | 13.4559% | 7.2066%          | -       |            1953 |             549 |

Разница лучшего embedding-варианта относительно baseline v2:

- FAR delta: -0.0431 п.п.
- FRR delta: +10.9314 п.п.
- Balanced error delta: +5.4441 п.п.
- False accepts delta: -88
- False rejects delta: +446

## 7. Сформированные артефакты

- `D:\Projects\keystroke_auth\models\embedding_model\embedding_classifier.keras`
- `D:\Projects\keystroke_auth\models\embedding_model\encoder.keras`
- `D:\Projects\keystroke_auth\models\embedding_model\embedding_scaler.pkl`
- `D:\Projects\keystroke_auth\models\embedding_model\embedding_label_encoder.pkl`
- `D:\Projects\keystroke_auth\users\embedding_model\user_templates_embedding.json`
- `D:\Projects\keystroke_auth\users\embedding_model\user_thresholds_embedding.json`
- `D:\Projects\keystroke_auth\reports\embedding_model\embedding_classifier_metrics.csv`
- `D:\Projects\keystroke_auth\reports\embedding_model\embedding_distance_diagnostics.csv`
- `D:\Projects\keystroke_auth\reports\embedding_model\embedding_threshold_policy.csv`
- `D:\Projects\keystroke_auth\reports\embedding_model\embedding_enrollment_size_experiment.csv`
- `D:\Projects\keystroke_auth\reports\embedding_model\embedding_vs_softmax_comparison.csv`

## 8. Итоговый вывод

Embedding-based verification реализована и проверена на полном цикле: encoder, user templates, distance diagnostics, threshold policy, CLI-аутентификация и enrollment-size experiment.

Лучший embedding-вариант использует cosine distance и per-user threshold policy. Однако при FAR около 1% он даёт существенно более высокий FRR, чем softmax baseline v2.

Основная причина: encoder обучался как softmax-классификатор через cross-entropy loss. Такое обучение хорошо решает задачу закрытой классификации пользователей, но не оптимизирует embedding-пространство напрямую под distance-based verification.

Практический вывод:

- softmax baseline v2 остаётся основной рабочей моделью текущего этапа;
- embedding-подход следует развивать через Siamese / contrastive / triplet loss или metric-learning fine-tuning;
- результаты stage 6 можно использовать как отрицательный, но методологически значимый эксперимент для статьи.
