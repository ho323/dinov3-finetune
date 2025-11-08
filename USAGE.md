# DINOv3 Fine-tuning 사용법

이 프로젝트는 DINOv2/DINOv3를 LoRA를 사용하여 이미지 세그멘테이션 태스크로 파인튜닝하는 프로젝트입니다.

## 설치

```bash
# conda 환경 생성
conda create --name dino python=3.11
conda activate dino

# 패키지 설치
pip install -e .
```

## 데이터셋 준비

### 1. Pascal VOC 또는 ADE20k

기존 데이터셋을 사용하는 경우 자동으로 다운로드됩니다.

### 2. OpenFake 데이터셋

OpenFake 데이터셋은 Parquet 파일 형식으로 저장되어 있어야 합니다.

**데이터셋 구조:**
- 데이터셋 디렉토리: `D:\dataset\OpenFake\data` (기본값)
- 각 Parquet 파일은 `image` 컬럼을 포함해야 함
- `image` 컬럼은 `{'bytes': b'...'}` 형식의 딕셔너리

**예시:**
```
D:\dataset\OpenFake\data\
  ├── batch_00000.parquet
  ├── batch_00001.parquet
  └── ...
```

각 Parquet 파일의 `image` 컬럼은 다음과 같은 형식입니다:
```python
{
    'bytes': b'\xff\xd8\xff\xe0...'  # JPEG 이미지 바이트
}
```

## 학습 실행

### 기본 사용법

#### OpenFake 데이터셋으로 DINOv3 학습

```bash
python main.py \
    --exp_name openfake_dinov3_large \
    --dataset openfake \
    --dataset_root "D:\dataset\OpenFake" \
    --size large \
    --dino_type dinov3 \
    --img_dim 512 512 \
    --epochs 100 \
    --use_lora \
    --batch_size 32 \
    --lr 3e-3


python main.py --exp_name openfake_dinov3_small --dataset openfake --dataset_root "D:\dataset\OpenFake" --size small --dino_type dinov3 --checkpoint_path "C:\Users\ho\Desktop\공모전\dinov3-finetune\dinov3\checkpoints\dinov3_vits16_pretrain_lvd1689m-08c60483.pth" --img_dim 512 512 --epochs 10 --use_lora --batch_size 32
```

#### Pascal VOC 데이터셋으로 학습

```bash
python main.py \
    --exp_name base_voc \
    --dataset voc \
    --size base \
    --dino_type dinov3 \
    --img_dim 308 308 \
    --epochs 50 \
    --use_lora \
    --use_fpn
```

#### ADE20k 데이터셋으로 학습

```bash
python main.py \
    --exp_name large_ade20k \
    --dataset ade20k \
    --size large \
    --dino_type dinov3 \
    --img_dim 490 490 \
    --epochs 100 \
    --use_lora
```

### 주요 파라미터

#### 필수 파라미터
- `--exp_name`: 실험 이름 (결과 저장 시 사용)
- `--dataset`: 데이터셋 이름 (`voc`, `ade20k`, `openfake`)
- `--size`: 모델 크기 (`small`, `base`, `large`, `giant`, `huge`)
- `--dino_type`: DINO 버전 (`dinov2`, `dinov3`)

#### OpenFake 전용 파라미터
- `--dataset_root`: OpenFake 데이터셋 경로 (기본값: `D:\dataset\OpenFake\data`)

#### 학습 파라미터
- `--epochs`: 학습 에포크 수 (기본값: 100)
- `--batch_size`: 배치 크기 (기본값: 64)
- `--lr`: 학습률 (기본값: 3e-3)
- `--min_lr`: 최소 학습률 (기본값: 3e-5)
- `--warmup_epochs`: 워밍업 에포크 수 (기본값: 20)
- `--weight_decay`: 가중치 감쇠 (기본값: 1e-2)

#### 모델 파라미터
- `--use_lora`: LoRA 사용 여부 (플래그)
- `--r`: LoRA rank 파라미터 (기본값: 3)
- `--use_fpn`: FPN 디코더 사용 여부 (플래그)
- `--img_dim`: 이미지 크기 (height width, 기본값: 490 490)
- `--lora_weights`: 사전 학습된 LoRA 가중치 경로

#### 기타
- `--debug`: 디버그 모드 (시각화 출력)

### 이미지 크기 주의사항

이미지 크기는 패치 크기의 배수여야 합니다:
- DINOv3: 패치 크기 16
- DINOv2: 패치 크기 14

예를 들어, DINOv3의 경우 512x512, 480x480, 496x496 등이 가능합니다.

## 결과 저장

학습 결과는 다음 위치에 저장됩니다:
- 모델 가중치: `output/{exp_name}.pt`
- 메트릭: `output/{exp_name}_metrics.json`
- 중간 체크포인트: `output/{exp_name}_e{epoch}.pt` (5 에포크마다)

## 사전 학습된 모델 사용

```bash
python main.py \
    --exp_name continue_training \
    --dataset voc \
    --lora_weights output/dinov3/large_voc_lora.pt \
    --epochs 50 \
    --use_lora
```

## OpenFake 데이터셋 특이사항

OpenFake 데이터셋은 세그멘테이션 마스크가 없으므로, 자동으로 더미 마스크가 생성됩니다:
- 전체 이미지를 클래스 1로 설정
- 실제 세그멘테이션 태스크가 필요한 경우, 마스크 데이터를 추가해야 합니다

데이터셋은 자동으로 80% train / 20% validation으로 분할됩니다.

## 문제 해결

### 메모리 부족
- `--batch_size`를 줄이세요
- `--img_dim`을 줄이세요

### 학습 속도가 느림
- `num_workers`를 조정하세요 (data.py에서)
- GPU 메모리에 맞게 배치 크기를 조정하세요

### 데이터셋 경로 오류
- `--dataset_root`로 올바른 경로를 지정하세요
- Parquet 파일이 해당 디렉토리에 있는지 확인하세요

