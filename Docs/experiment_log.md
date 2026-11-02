# 실험 과정

HCP task classification으로 학습한 fMRI encoder를 NSD caption decoding에 연결한 과정이다.

## 1. 데이터 흐름

전처리 실습과 모델 학습은 별도로 진행했다.

```text
raw HCP / NSD data → 전처리 → GLM → 직접 추출한 betamap → ablation

제공된 HCP betamap → 19개 task condition 분류 → encoder 가중치 저장
                                                     ↓
제공된 NSD betamap + COCO 캡션 → encoder → Bridge → GPT-2 → 캡션 생성
```

### 전처리

slice-timing correction, rigid-body motion correction, spatial smoothing을 적용했다. 메모리 제약으로 일부 전처리 과정은 생략했다(ipynb 참고). 이후 GLM으로 반응 크기를 추정하고 마스크를 적용해 벡터로 저장했다.

| 항목 | HCP | NSD |
|---|---|---|
| 피험자 | 100206 | sub-01 |
| TR | 0.72초 | 1.6초 |
| 처리 run 수 | 12 | 15 |
| 저장 단위 | run 내 task condition | run 내 이미지 ID |
| 최종 배열 | `(30, 182453)` | `(133, 292796)` |

## 2. HCP classification

제공된 betamap 1,520개에서 19개 task condition을 분류했다. 각 벡터를 길이 100,000으로 pooling하고 sample 단위로 train / validation / test를 8:1:1로 분할했다.

```text
betamap → Conv1d block × 4 → pooling → FC → feature vector (768) → classification head
```

Adam, learning rate `1e-4`, weight decay `1e-5`, batch size 64로 35 epochs 학습했다.

### Loss curve

![HCP classification loss](Images/encoder-loss.png)

| epoch | train loss | validation loss |
|---|---|---|
| 5 | 0.3713 | 0.6386 |
| 15 | 0.0155 | 0.4836 |
| 25 | 0.0046 | 0.5241 |
| 35 | 0.0020 | 0.5422 |

분류 결과와 직접 전처리한 HCP data의 ablation은 [Classification Results](../Results/Classification/Results.md)에 정리했다.

## 3. NSD caption decoding

HCP classification head를 제거하고 저장한 encoder를 불러왔다. NSD betamap과 COCO 캡션 7,500쌍을 train 6,000 / validation 750 / test 750으로 나눴다. 각 sample에는 이미지의 캡션 중 하나를 dataset 생성 시 선택했다.

```text
betamap (100,000)
 → pretrained encoder
 → feature vector (768)
 → Bridge: Linear + Dropout
 → brain-prefix 3개 (각 768차원)
 → frozen GPT-2
 → 생성 캡션
```

GPT-2 가중치는 고정하고 encoder와 Bridge를 학습했다. 학습 시에는 prefix와 정답 캡션 embedding을 연결하며, 생성 시에는 prefix만 입력한다.

### Loss curve

![NSD caption decoding loss](Images/btt-loss.png)

Adam, learning rate `1e-4`, batch size 64로 25 epochs 학습했다.

| epoch | train loss | validation loss |
|---|---|---|
| 5 | 1.4722 | 1.4121 |
| 10 | 1.4230 | 1.3438 |
| 15 | 1.3944 | 1.3147 |
| 20 | 1.3721 | 1.3091 |
| 25 | 1.3495 | 1.2887 |

실제 생성된 캡션들은 [Caption Examples](../Results/Caption_examples/example.md)에 정리했다.

## 관련 노트북

- [01 HCP preprocessing](../Notebooks/01_HCP_preprocessing.ipynb)
- [02 NSD preprocessing](../Notebooks/02_NSD_preprocessing.ipynb)
- [03 HCP classification](../Notebooks/03_HCP_classification.ipynb)
- [04 NSD caption decoding](../Notebooks/04_NSD_caption_decoding.ipynb)
