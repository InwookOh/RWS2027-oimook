# IRRA-QueryBalanced-ViT-L14

이 프로젝트는 공식 OpenAI CLIP ViT-L/14 가중치로 시작하는 QueryBalanced 검색 모델을 학습합니다. 기존 SDM+MAP 목적 함수와 60 Epoch FP32 설정을 사용합니다.

저장소 루트에서 다음 명령을 실행하세요.

```bash
bash scripts/train_pipeline.sh download
bash scripts/train_pipeline.sh smoke
bash scripts/train_pipeline.sh querybalanced
```

파이프라인은 공식 가중치를 검증하고 데이터셋 및 CUDA 환경을 확인한 뒤 유휴 GPU 하나를 선택해 새로운 Run 디렉터리를 만듭니다. 학습 완료 후 생성된 Epoch 2 Checkpoint와 메타데이터는 FrozenMultiScaleLocalAttr 학습에 자동으로 사용됩니다.

설치, 데이터셋 구조, GPU 및 데이터 경로 선택, 두 모델의 전체 학습 절차는 저장소 [학습 안내](../README.md)를 참고하세요. 학습 설정은 `configs/train_fp32_micro4.json`에 그대로 있습니다.
