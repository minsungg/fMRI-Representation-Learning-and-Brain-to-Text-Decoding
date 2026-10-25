# fMRI Representation Learning and Brain-to-Text Decoding

fMRI 뇌 활동 데이터로 인지 과제를 분류하고, 피험자가 본 이미지의 내용을 문장으로 생성하는 수업 프로젝트입니다.

- **기간:** 2026.04–2026.06
- **수행자:** 유민성(개인)
- **담당 작업:** HCP·NSD 전처리, fMRI 인코더 설계·학습, GPT-2 기반 캡션 생성 및 결과 분석

## 주요 내용

1. 원시 fMRI 전처리 및 GLM 기반 beta-map 추출
2. 1D CNN으로 HCP 19개 과제 분류
3. 인코더–Bridge–GPT-2를 연결한 NSD 캡션 생성

모델 학습에는 수업에서 제공한 beta-map을 사용했으며, 직접 전처리한 데이터는 ablation에 활용했습니다.

구현 과정과 실행 결과는 각 노트북에 정리했습니다.
