# fMRI Representation Learning and Brain-to-Text Decoding

fMRI 뇌 활동 data로 인지 과제를 분류하고, 피험자가 본 이미지의 내용을 문장으로 생성하는 수업 프로젝트다. HCP betamap으로 학습한 encoder를 GPT-2에 연결해 NSD 캡션 생성으로 확장했다.

![Title](./)

- **기간:** 2026.04–2026.06
- **수행자:** 유민성(개인)
- **담당 작업:** HCP·NSD 전처리, fMRI encoder 설계·학습, GPT-2 기반 캡션 생성 및 결과 분석

## 주요 내용

### 1. HCP·NSD 전처리

raw fMRI data에 slice-timing correction, motion correction, spatial smoothing을 적용하고 GLM으로 task condition 및 이미지별 betamap을 추출했다. NSD는 이미지 ID를 통해 COCO 캡션과 연결했다.

### 2. HCP task classification

1D CNN encoder와 classification head로 HCP 19개 task condition을 분류했다. sample 단위 무작위 분할에서 **test Macro-F1 0.8667**을 기록하고, encoder 가중치를 저장했다.

### 3. NSD caption decoding

학습한 encoder의 feature vector를 Bridge로 변환해 GPT-2에 brain-prefix로 입력했다. GPT-2 가중치가 frozen인 상태로 encoder와 Bridge network만 학습했다. 생성 캡션과 ablation, 직접 전처리한 data의 결과를 비교했다.

모델 학습에는 수업에서 제공한 betamap을 사용했다. 직접 전처리한 data는 별도의 입력 테스트에 활용했다.

## 구성

| 경로 | 내용 |
|---|---|
| [Notebooks](Notebooks/) | 전처리부터 분류·캡션 생성까지의 코드와 실행 결과 |
| [Results](Results/) | 결과 이미지 및 생성 예시 |
| [Docs](Docs/) | 실험 과정과 한계 정리 |

### 노트북

1. [HCP preprocessing](Notebooks/01_HCP_preprocessing.ipynb)
2. [NSD preprocessing](Notebooks/02_NSD_preprocessing.ipynb)
3. [HCP classification](Notebooks/03_HCP_classification.ipynb)
4. [NSD caption decoding](Notebooks/04_NSD_caption_decoding.ipynb)
