# FrozenMultiScaleLocalAttr 추론

## 1. GitHub Clone 및 환경 설치

```bash
cd third_party/irra
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
mkdir -p data
curl -fL https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz \
  -o data/bpe_simple_vocab_16e6.txt.gz
```

## 2. Checkpoint 저장 위치

파일명은 `epoch_002.pth`입니다. 전달받은 전체 Checkpoint를 아래 위치에 복사합니다. 별도의 ViT-L/14 Weight는 필요하지 않습니다.

```bash
mkdir -p experiments/frozen_localattr_e5_20261013_041809_KST
cp /전달받은/경로/epoch_002.pth \
  experiments/frozen_localattr_e5_20261013_041809_KST/epoch_002.pth
```

## 3. Inference 실행

`--data-root`는 `annotations/task2/val/{queries.csv,ids.csv,gt.csv}`와 해당 Gallery 이미지가 있는 UPAR 데이터셋 루트입니다.

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
  --checkpoint experiments/frozen_localattr_e5_20261013_041809_KST/epoch_002.pth \
  --data-root /데이터셋/UPAR \
  --device cuda:0 \
  --batch-size 4 \
  --output-dir outputs/frozen_localattr_epoch002
```

GPU 대신 CPU를 사용하려면 `--device cpu`로 실행합니다. 결과 디렉터리에 Codabench와 같은 정렬 의미를 갖는 `distances.npy`(Query × Gallery, 작은 값 우선), 지표 `metrics.json`, 공식 Query/Gallery 순서 `matrix_order.json`이 저장됩니다.
