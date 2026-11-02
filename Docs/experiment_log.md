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

### Encoder 구조 변경

초기에는 MLP를 사용했지만 training loss가 1.0 아래로 내려가지 않았다. 이에 평탄화한 betamap에서 인접한 값들의 패턴을 학습하는 1D CNN으로 변경했다. 긴 입력을 순차적으로 처리하는 RNN의 실행 시간과, 적은 학습 data에서 Transformer를 학습하는 부담을 고려해 CNN을 선택했다.

두 번째 구성은 convolution layer 3개와 FC layer 2개로 256차원 feature vector를 만드는 구조였다. HCP 분류에서는 validation 성능이 나왔지만, NSD 캡션 생성에 재사용했을 때 반복적인 문장과 자극에 맞지 않는 출력이 나타났다.

이후 convolution layer를 4개로 늘리고 ReLU를 GELU로 변경했으며 dropout을 추가했다. convolution block의 dropout rate는 0.1로 설정했다. feature dimension도 768로 확대했으며, 이 변경은 아래 NSD 실험과 함께 조정했다.

classification head는 단일 linear layer로 구성하고 encoder와 분리했다. HCP 분류 이후에는 head를 제거하고 feature vector를 NSD에 재사용하기 위한 구성이다.

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

### Feature dimension과 prefix 조정

초기에는 256차원 feature vector를 여러 brain-prefix로 변환했다. prefix 수를 늘리면 training loss는 낮아졌지만, 실제 생성에서는 자극과 관계없는 문장이나 반복 표현이 나타났다. 학습 loss가 낮아지는 것과 새로운 입력에서 캡션이 잘 생성되는 것은 달랐다.

이 과정에서 prefix 수뿐 아니라 Bridge에 전달되는 feature vector의 폭을 함께 조정했다. 출력 벡터를 크게 만들더라도 그 바탕이 되는 256차원 표현은 그대로였기 때문이다. feature dimension을 256에서 768로 늘리고 prefix는 최종 3개로 줄였다. 당시 실험에서는 feature dimension 확대가 생성 결과 개선의 주요 전환점이었다.

prefix를 더 늘렸을 때는 성능 향상이 크지 않은 반면 전체 학습 시간이 늘어 최종 3개를 사용했다. 중간 구성별 비교 수치는 별도로 남아 있지 않아, 여기에는 변경 과정만 정리했다.

### Bridge와 학습 대상 선택

Bridge는 계산량과 구조를 단순하게 유지하기 위해 단일 linear layer로 구성했다. 별도의 자연어 지시문 없이 brain-prefix를 입력 앞에 배치해 캡션 생성을 유도했다.

GPT-2 전체를 학습하는 계산 부담을 줄이기 위해 가중치를 고정했다. 반면 encoder는 HCP 분류에서 학습한 표현을 NSD에 맞게 조정할 수 있도록 Bridge와 함께 학습했다. 단일 linear layer인 Bridge에 변환을 모두 맡기기보다, 그 앞의 feature vector도 함께 조정하는 방식이다.

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
