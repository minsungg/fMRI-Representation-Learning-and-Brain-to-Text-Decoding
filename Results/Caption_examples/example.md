# NSD 캡션 생성 예시와 ablation

4번째 ipynb 파일에서 진행된 캡션 생성의 예시를 나열한다. 모델 구조와 학습 과정은 [실험 기록](../../Docs/experiment_log.md#3-nsd-caption-decoding)에 정리했다.

## 1. Test data의 생성 결과

| sample | reference | generated caption |
|---|---|---|
| 1 | several red ripe apples in two large wooden boxes | A large bowl of rice and a large bowl of noodles. |
| 2 | A woman sitting at a table with plates of food. | A woman is sitting on a bench with her hands on her hips. |
| 3 | Room with wood covered wall and floor with two beds. | A large room with a bed and a desk. |
| 4 | a green field that has a man with a kite | A man standing on a tree. |
| 6 | A seagull rests on a concrete wall, with the ocean and a lighthouse in the background. | A man is standing on a tree in the middle of the road. |
| 7 | A zebra is standing in a field and looking right at the camera. | A man is standing on a tree in the middle of the forest. |

## 2. ablation

실제 betamap 대신 zero 벡터 하나와 random 벡터 세 개를 입력했다.

| 입력 | generated caption |
|---|---|
| zero | A man walks up to a tree and sees a tree. |
| random 1 | A man walks up to a tree with his hands on a tree. |
| random 2 | A man walks up to a tree with his hands on a tree. |
| random 3 | A man walks up to a tree with his hands on a tree. |

## 3. 직접 전처리한 NSD data

02 노트북에서 추출한 betamap을 학습된 모델에 입력했다. 서로 다른 자극에서도 유사한 문장이 반복됐다.

| reference | generated caption |
|---|---|
| Skiers and snowboarders going down a run on a sunny day. | A man standing on a beach with his head covered in sand. |
| A pizza on a plate with a fork and knife. | A man is standing on a beach with his head covered in sand. |
| A large dog wearing an orange safety vest. | A man standing on a beach with his hands on his head. |
| A cat rclining in a red leather easy chair | A man standing on a beach with his head covered in sand. |

---

[실험 과정](../../Docs/experiment_log.md) · [Classification Results](../Classification/Results.md) · [04 NSD caption decoding 노트북](../../Notebooks/04_NSD_caption_decoding.ipynb)
