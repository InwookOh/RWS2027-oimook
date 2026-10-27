# IRRA-QueryBalanced-ViT-L14 Learning Rate 분석

분석 범위는 기존 `IRRA-QueryBalanced-ViT-L14` 프로젝트의 실제 실행 Config, 코드, 데이터 감사 파일, Training Log, Validation 기록이다. 코드, Config, Checkpoint는 수정하지 않았고 추가 학습도 하지 않았다.

## 핵심 판단

Unseen Domain 일반화를 우선한다면, 현재 기록만으로 **전체 LR을 올리기보다 pretrained CLIP backbone의 LR을 낮추는 실험이 더 합리적**이다. 우선 `CLIP Image/Text Encoder = baseline의 0.25~0.5배`, 새 IRRA MAP/CMT 모듈은 baseline LR을 유지하는 차등 LR을 권한다. 전체 LR을 일괄 0.5배로 낮추는 비교군도 두는 편이 좋다.

단, “현재 LR이 너무 높다”는 인과관계가 증명된 것은 아니다. 확인된 실행에서는 Validation 지표가 Epoch 3까지 개선된 뒤 Epoch 4에 하락했고, 동시에 Train Loss는 계속 감소했다. 이는 조기 과적합 또는 일반화 간극의 **징후와 일치**하지만, LR, 누적 학습량, QueryBalanced 반복 노출, 도메인 차이 중 무엇이 원인인지는 이 로그만으로 분리할 수 없다. 또 사용자 설명의 “최고 Epoch 2”는 이 프로젝트에서 찾은 단일 실행 기록과 다르다. 그 실행에서는 기록된 모든 Validation 지표가 Epoch 3에서 가장 높았다.

## 확인한 실행과 데이터

프로젝트에서 찾은 실행은 `experiments/full_20261012_171058_KST_d66f37` 하나다. 실제 실행 Config는 `configs/train_fp32_micro4.json`이며, `data_root`는 프로젝트의 `data/`다. 학습에 사용한 것은 **Train split만**이고, 공개 Validation split은 별도 평가에 사용했다. Train은 97,669장/7,204개 attribute combination, Validation은 33,407장/3,462개 query다. 감사 결과 Train/Validation의 이미지 ID와 resolved path 중복은 각각 0건이다.

Train domain 구성은 PA100k 79,001장(약 80.9%), Market1501 10,000장(10.2%), PETA 8,668장(8.9%)이다. Validation은 Market1501 16,458장(49.3%), PA100k 9,986장(29.9%), PETA 6,963장(20.8%)이다. 따라서 공개 Validation은 Train과 완전히 별개 이미지지만, 세 데이터 출처가 겹치는 pooled validation이지 숨겨진 Unseen Domain 검증은 아니다. 픽셀 내용 수준의 중복 검사는 수행되지 않았다.

QueryBalanced sampler는 Train의 정확한 40-bit vector별 빈도를 세고, 각 이미지에 `f(q)^(-0.5)` 가중치를 준 뒤 replacement sampling으로 매 Epoch 97,669개를 뽑는다. 조합별 총 sampling 질량은 대략 `sqrt(f(q))`에 비례하므로 일반 무작위 sampling보다 희귀 조합이 상대적으로 더 자주 선택되지만, 조합 빈도가 완전히 같아지는 방식은 아니다. 이로 인한 희귀 조합 과적합은 가능한 가설이나, 현재 로그에는 조합별 Validation 분석이 없어 확인되지 않았다. Domain 빈도를 직접 맞추는 sampler도 아니다.

## 실제 모델·Optimizer·Scheduler 설정

- Backbone은 OpenAI CLIP ViT-L/14다. SDM은 global Image/Text feature를 함께 정렬한다. MAP는 masked attribute token 예측이며, IRRA의 cross-attention, 4-layer cross-modal transformer(CMT), MLM head를 사용한다.
- 학습 시 CLIP Image Encoder와 Text Encoder, CLIP projection, CMT/MAP 모듈을 freeze하지 않았다. `build_optimizer`는 `requires_grad`인 전체 파라미터를 **단일 Adam parameter group**에 넣는다. 별도의 Backbone/Text/Image/Projection/MAP LR은 없다. `logit_scale`은 temperature 0.02의 고정 buffer라 학습 대상이 아니다.
- Adam: `lr=1e-5`, `betas=(0.9, 0.999)`, `eps=1e-3`, `weight_decay=4e-5`; bias와 normalization 파라미터에도 같은 weight decay가 적용된다. AdamW가 아니라 `torch.optim.Adam`이다.
- Scheduler는 60 Epoch 기준 **Epoch 단위 Linear Warm-up + Cosine Decay**, `target_lr=0`다. Scheduler는 각 Epoch의 Validation이 끝난 뒤 한 번 step하며, 한 Epoch 안에서는 LR이 일정하다. 설정의 `milestones=[20,50]`, `gamma=0.1`, `power=0.9`는 cosine 분기에서 사용되지 않는다. 이들은 다른 scheduler mode에서만 작동한다.
- Warm-up은 5 Epoch, factor 0.1에서 시작한다. PyTorch 초기 scheduler step 및 실제 로그를 반영하면 Epoch 1~5의 LR은 최대 LR의 10%, 28%, 46%, 64%, 82%다. Epoch 6부터 cosine 구간이며 Epoch 6 시작에서 최대 LR에 도달한다.
- Batch size는 effective/global 32, micro-batch 4 × 8 chunk로 gradient cache를 계산하고 **각 DataLoader batch에서 optimizer update는 1회** 한다. 이는 8개의 별도 optimizer step이 아니다. 97,669개 sampling draw와 batch 32로 Epoch당 3,053 update(마지막 batch 5개)가 발생한다.
- 이 baseline은 DDP가 아니다. 실행 환경은 `WORLD_SIZE=1`이고 학습 코드는 `WORLD_SIZE != 1`이면 실패하도록 되어 있다. 따라서 실제 Epoch당 step은 3,053이며 DDP rank 수를 곱하거나 나누지 않는다. 60 Epoch를 완주하면 예정 step은 183,180, Warm-up 5 Epoch는 15,265 step이다.

산식은 warm-up에서 `lr = base_lr * (0.1 * (1 - last_epoch/5) + last_epoch/5)`, 이후 cosine에서 `lr = base_lr * 0.5 * (1 + cos(pi * (last_epoch-5)/(60-5)))`다. 코드는 [lr_scheduler.py](../solver/lr_scheduler.py), Optimizer 및 step 순서는 [training.py](../upar/training.py), 실제 값은 [train_fp32_micro4.json](../configs/train_fp32_micro4.json)에 있다.

## Epoch별 LR과 실제 업데이트

아래 `Epoch LR`은 해당 Epoch 전체의 optimizer update에 적용된 값이다. `다음 LR`은 Validation 후 `scheduler.step()`을 한 뒤 Checkpoint 저장 시 optimizer에 설정되어 있는 다음 Epoch 값이다. Checkpoint는 plain model `state_dict`여서 optimizer/scheduler/LR 자체는 저장하지 않는다.

| Epoch | Epoch LR | 최대 LR 대비 | 다음 LR (저장 시 scheduler 값) | Warm-up | 예정 optimizer step |
|---:|---:|---:|---:|:---:|---:|
| 1 | 1.000000e-6 | 10.000% | 2.800000e-6 | 예 | 3,053 |
| 2 | 2.800000e-6 | 28.000% | 4.600000e-6 | 예 | 3,053 |
| 3 | 4.600000e-6 | 46.000% | 6.400000e-6 | 예 | 3,053 |
| 4 | 6.400000e-6 | 64.000% | 8.200000e-6 | 예 | 3,053 |
| 5 | 8.200000e-6 | 82.000% | 1.000000e-5 | 예 | 3,053 |
| 6 | 1.000000e-5 | 100.000% | 9.991846e-6 | 아니오 | 3,053 |
| 7 | 9.991846e-6 | 99.918% | 9.967409e-6 | 아니오 | 3,053 |
| 8 | 9.967409e-6 | 99.674% | 9.926769e-6 | 아니오 | 3,053 |
| 9 | 9.926769e-6 | 99.268% | 9.870060e-6 | 아니오 | 3,053 |
| 10 | 9.870060e-6 | 98.701% | 9.797465e-6 | 아니오 | 3,053 |

따라서 **Epoch 2가 최고라는 가정이 맞는 다른 실행이라면** 그 Checkpoint는 최대 LR의 28%만 적용받았을 것이다. 이 실행에서 실제 최고인 Epoch 3은 최대 LR의 46%다. 둘 다 최고 성능이 LR peak보다 먼저 나왔다는 점은 조기 선택의 근거지만, 그 자체로 높은 LR이 원인이라는 증거는 아니다.

저장 시점 관련 주의: Epoch 1 모델 가중치는 Epoch 1 동안 `1.0e-6`으로 학습된 결과다. Validation 후 scheduler를 먼저 움직인 뒤 저장하므로 그 직후 optimizer LR은 `2.8e-6`이다. CSV의 `lr`과 `next_lr`가 이 차이를 명시한다. `epoch_001.pth` 파일 내부에 Epoch 또는 LR 메타데이터가 있는 것은 아니다.

## 실제 Loss 및 Validation 기록

| 완료 Epoch | Global step | Train Loss | SDM Loss | MAP Loss | LR | mADM | mAP | Rank-1 | Rank-5 | Rank-10 | mINP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3,053 | 12.0575 | 4.4021 | 7.6553 | 1.0e-6 | 68.551% | 27.788% | 25.794% | 51.098% | 61.612% | 18.051% |
| 2 | 6,106 | 5.1332 | 2.3585 | 2.7747 | 2.8e-6 | 68.921% | 27.867% | 26.054% | 50.635% | 60.630% | 18.366% |
| 3 | 9,159 | 2.6338 | 1.9637 | 0.6701 | 4.6e-6 | **69.248%** | **29.077%** | **27.325%** | **51.820%** | **62.305%** | **19.059%** |
| 4 | 12,212 | 2.0156 | 1.7652 | 0.2504 | 6.4e-6 | 67.830% | 28.178% | 25.737% | 51.502% | 61.612% | 18.408% |

사실로 확인되는 내용:

1. Epoch 2에서 Epoch 4까지 Train Loss, SDM Loss, MAP Loss 모두 계속 감소했다.
2. Validation의 모든 기록 지표가 Epoch 3에서 최고였으며, Epoch 4에서 mADM은 1.418%p, mAP는 0.899%p, Rank-1은 1.589%p 내려갔다. Rank-5/10은 Epoch 2보다 Epoch 4가 높아 모든 metric이 일관되게 나빠진 것은 아니다.
3. Validation은 Epoch 1→3 동안 개선되는 동안 LR도 상승했다. 하락은 LR이 4.6e-6에서 6.4e-6으로 오른 Epoch 3→4 구간에서 한 번 관측됐다. 이 한 구간만으로 특정 LR 임계값을 정할 수 없다.
4. 실행은 Epoch 4 평가/Checkpoint까지 완료했고 Epoch 5의 2,727/3,053 batch(global step 14,939)에서 graceful stop됐다. Epoch 5 평균 Train Loss, Validation, Checkpoint는 없다. 따라서 “첫 5 Epoch 결과”나 Epoch 5 성능은 보고할 수 없다.

이번 프로젝트에서 확인한 Validation 최고는 Epoch 2가 아니라 Epoch 3이다. Epoch 2가 최고였다는 관찰이 다른 실험에서 나온 것이라면 이 보고서의 수치와 섞어 해석하면 안 된다.

## 원인 가설 비교

| 가설 | 로그가 지지하는 부분 | 아직 확인되지 않은 부분 |
|---|---|---|
| 현재 LR이 높아 CLIP 표현을 훼손 | Train Loss가 빠르게 내려가고 Epoch 4에서 pooled validation 일부 지표가 후퇴했다. 모든 pretrained encoder와 projection을 같은 LR로 업데이트한다. Unseen Domain 우선이라면 backbone을 보수적으로 업데이트하는 검증은 타당하다. | Pretrained 대비 파라미터 거리, Image/Text feature drift, zero-shot retention, 도메인별 성능을 측정하지 않았다. LR 상승은 Epoch 1~3의 validation 개선과도 함께 나타났다. 손상·catastrophic forgetting을 직접 확인한 것은 아니다. |
| 현재 LR이 너무 낮음 | Warm-up 중이라 Epoch 2는 최대 LR의 28%, Epoch 3은 46%다. 제한된 초기 Epoch에서 더 빠른 적응을 시도할 여지는 있다. | Train 손실은 이미 빠르게 감소했고 E3에서 모든 validation 지표가 최고다. underfitting을 보여주는 기록은 없다. LR 증가가 Unseen Domain에 이롭다는 증거도 없다. |
| LR보다 데이터·목표·학습기간 요인이 큼 | 4 Epoch 동안 Train만으로 97,669 draw/epoch를 반복한다. Domain 구성은 크게 불균형하고 QueryBalanced는 domain을 직접 균형화하지 않는다. Image/Text backbone을 동시에 학습하며 SDM+MAP를 합산한다. | Domain/attribute-combination별 학습 곡선, encoder drift, seed 반복을 보지 않았다. 어느 요인의 기여도도 분리되지 않았다. |

이 실행은 pretrained CLIP Feature가 실제로 빠르게 손상됐다고 입증하지 않는다. MAP train loss가 낮아진 것은 masked token 학습이 빠르게 맞춰졌다는 뜻이지, Unseen Domain의 text-image alignment가 보존 또는 붕괴했다는 직접 측정이 아니다. 마찬가지로 희귀 조합의 replacement 노출은 과적합 위험을 높일 수 있지만, 현재 지표로는 특정 조합 memorization을 확인할 수 없다.

CLIP fine-tuning이 distribution shift 성능을 낮출 수 있다는 선행 연구는 일반적인 위험의 근거로 참고할 수 있다. WiSE-FT는 zero-shot 모델과 fine-tuned 모델의 weight interpolation으로 robustness를 회복할 수 있음을 보고했고, CLIPood는 fine-tuned checkpoint가 pretrained 모델에서 멀어질 때 OOD generalization이 나빠질 수 있는 문제를 다뤘다. 이 결과들은 이 UPAR 설정이나 특정 LR 배율이 개선된다는 보장은 아니다. [WiSE-FT, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Wortsman_Robust_Fine-Tuning_of_Zero-Shot_Models_CVPR_2022_paper.html), [CLIPood, ICML 2023](https://proceedings.mlr.press/v202/shu23a.html).

## 다음 LR 실험 제안

기존 60-Epoch cosine 및 5-Epoch warm-up의 **형태와 총 길이를 먼저 고정**하고, LR multiplier만 바꿔 비교한다. 5 Epoch에서 조기 종료해도 총 Epoch를 5로 바꿔 schedule을 압축하지 않는다. 같은 training seed, sampler 규칙, effective batch, loss를 유지하고 별도 seed 반복을 추가하면 변동성을 판단하기 좋다.

| 배율 | Base LR | Epoch 2 LR | Peak LR (Epoch 6) | 기대/위험 |
|---:|---:|---:|---:|---|
| 0.25× | 2.5e-6 | 0.7e-6 | 2.5e-6 | pretrained backbone을 가장 강하게 보존할 후보. 적응 속도가 너무 느려질 수 있음. |
| 0.5× | 5.0e-6 | 1.4e-6 | 5.0e-6 | 전체 LR 감소의 우선 비교군. 안정성을 높일 수 있지만 MAP/CMT 적응도 함께 늦어짐. |
| 1.0× | 1.0e-5 | 2.8e-6 | 1.0e-5 | 현재 baseline 비교군. |
| 2.0× | 2.0e-5 | 5.6e-6 | 2.0e-5 | Task 적응은 빨라질 수 있으나 현재 조기 validation 하락 및 OOD 우선 목표와 맞지 않는 위험이 커 낮은 우선순위. |

가장 추천하는 세 비교는 다음이다.

1. **차등 LR:** CLIP Image/Text backbone과 그 안의 projection을 0.25× 또는 0.5×로 낮추고, 새 IRRA CMT/MAP 파라미터는 1.0× 유지. pretrained 표현의 이동을 제한하면서 새 모듈의 학습 속도를 유지할 수 있어 가장 목표에 맞는다.
2. **전체 0.5×:** 모든 학습 파라미터를 같은 비율로 낮춰, uniform LR 감소 자체의 영향을 확인한다.
3. **전체 0.25×:** 더 강한 보존 방향 확인. 0.5×보다 task adaptation이 느린 위험이 있으므로 세 번째로 둔다.

현재 baseline 1.0×는 꼭 비교 기준으로 유지하고, 2.0×는 이들 실험이 underfit을 보일 때의 후속 진단 정도로 두는 편이 낫다. 초기 비교에서는 Image/Text backbone을 서로 다른 LR로 나누기보다 둘 모두 같은 낮은 multiplier를 먼저 적용해 원인을 덜 섞는다. Image/Text 둘 중 하나만 낮추는 비교는 후속 ablation으로 분리한다. Weight decay와 scheduler shape는 동시에 바꾸지 않는다. Scheduler LR은 각 group base LR에 같은 warm-up/cosine factor를 곱하도록 한다.

## Unseen Domain 평가 계획

공개 Validation은 Train과 경로가 겹치지 않지만 같은 세 benchmark 출처에 속해, hidden Unseen Domain 성능을 대신 보장하지 않는다. 다음 LODO는 hidden Test label을 사용하지 않는 proxy 평가로 제안한다.

| Fold | 학습에 사용할 Train domain | 독립 평가 |
|---|---|---|
| Market1501 hold-out | PA100k + PETA Train | Market1501 Validation |
| PA100k hold-out | Market1501 + PETA Train | PA100k Validation |
| PETA hold-out | Market1501 + PA100k Train | PETA Validation |

각 fold에서 hold-out domain의 어떤 이미지나 annotation도 training/sampling 통계에 포함하지 않는다. 나머지 두 domain의 Train split만 학습에 사용하고, LR 후보를 동일 seed/스케줄로 비교한다. pooled 평균뿐 아니라 domain별 mADM/mAP/Rank-1, 세 fold 평균과 최악 fold를 함께 본다. LR 선택에 이 Validation을 사용하면 해당 fold 결과는 hyperparameter selection용이므로 unbiased 최종 평가로 부르지 않는다. 세 데이터셋 LODO도 실제 hidden test 도메인을 완전히 대변한다는 보장은 없다.

## 결론 및 제한

질문에 대한 답은 **LR 감소 쪽이 더 합리적**이다. 다만 전체 LR을 무조건 크게 낮추기보다, 우선 pretrained CLIP backbone만 0.25~0.5×로 낮추고 신규 CMT/MAP는 baseline LR로 유지하는 차등 LR을 가장 먼저 검토한다. 전체 0.5×와 0.25×를 비교군으로 두고 LODO에서 선택한다. LR 2× 증가는 현재 데이터로 근거가 약하며 Unseen Domain 목표에서 위험이 더 크다.

현재 최고 검증 Epoch가 실제로는 3이고, 완결 로그가 Epoch 4까지만 있다는 점은 중요한 제한이다. Validation 감소는 한 번 관측됐고, hidden Test 성능, per-domain Validation curve, pretrained-vs-finetuned feature drift, 반복 seed가 없다. 그러므로 “LR이 너무 높다”는 진단은 확정이 아니라 검증할 우선 가설이며, 특정 배율이 성능을 개선한다고 단정할 수 없다.

## 확인한 로컬 근거

- Config: `configs/train_fp32_micro4.json`
- Optimizer / Epoch scheduler step / checkpoint 순서 / DDP 제한: `upar/training.py`
- Warm-up 및 cosine 계산: `solver/lr_scheduler.py`
- QueryBalanced 빈도 가중치와 replacement sampling: `upar/sampling.py`
- Train-only loader 구성: `upar/data.py`
- 실제 Validation 및 train loss 집계: `experiments/full_20261012_171058_KST_d66f37/metrics.csv`, `metrics.jsonl`
- 중단 시점 및 실제 batch/step: `experiments/full_20261012_171058_KST_d66f37/status.json`, `train.log`
- split/domain 통계 및 ID/path overlap: `experiments/full_20261012_171058_KST_d66f37/data_audit.json`
