# Finetuning DINOv3 with LoRA for OpenFake Dataset

This repository provides tools for finetuning DINOv3 (Siméoni et al., 2025) encoder weights using Low-Rank Adaptation (LoRA) for image segmentation tasks on the OpenFake dataset. LoRA enables efficient finetuning by adding a small set of trainable parameters between encoder blocks without modifying the original encoder weights.

## Features

- **DINOv3 Support**: Finetune DINOv3 encoders with LoRA for downstream tasks
- **Large Dataset Support**: Efficient lazy loading from Parquet files, enabling training on 2TB+ datasets
- **Memory Efficient**: Single-process data loading and index caching to prevent memory overflow
- **OpenFake Dataset**: Optimized for the OpenFake fake image detection dataset

## Setup

```bash
# Create conda environment
conda create --name dino python=3.11
conda activate dino

# Install package
pip install -e .
```

## OpenFake Dataset

The OpenFake dataset is a large-scale fake image detection dataset stored in Parquet format.

**Dataset Structure:**
```
D:\dataset\OpenFake\
  └── data\
      ├── batch_00000.parquet
      ├── batch_00001.parquet
      └── ...
```

**Parquet File Format:**
- `image`: JPEG image bytes stored as `{'bytes': b'...'}` dictionary
- `prompt`: Text prompt used for image generation
- `label`: `'real'` or `'fake'`
- `model`: Model name used for generation

**Dataset Statistics:**
- Training samples: ~635,000 (80% of parquet files)
- Validation samples: ~30,000 (20% of parquet files)
- Total dataset size: ~2TB
- Classes: 2 (real, fake)

## Usage

### Basic Training Command

```bash
python main.py \
    --exp_name openfake_dinov3_small \
    --dataset openfake \
    --dataset_root "D:\dataset\OpenFake" \
    --size small \
    --dino_type dinov3 \
    --checkpoint_path "path/to/dinov3_vits16_pretrain_lvd1689m-08c60483.pth" \
    --img_dim 512 512 \
    --epochs 100 \
    --use_lora \
    --batch_size 32 \
    --lr 3e-3
```

### Key Parameters

**Required:**
- `--exp_name`: Experiment name for saving results
- `--dataset`: Set to `openfake`
- `--dataset_root`: Root directory containing the `data/` folder with parquet files
- `--checkpoint_path`: Path to local DINOv3 checkpoint file (.pth)
- `--size`: Model size (`small`, `base`, `large`, `giant`, `huge`)
- `--dino_type`: Set to `dinov3`
- `--img_dim`: Image dimensions (height width). Must be divisible by 16 for DINOv3
- `--use_lora`: Flag to enable LoRA finetuning

**Optional:**
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 64)
- `--lr`: Learning rate (default: 3e-3)
- `--r`: LoRA rank parameter (default: 3)
- `--dinov3_repo_dir`: Local path to dinov3 repository (defaults to torch hub cache)
- `--use_fpn`: Use FPN decoder instead of linear decoder
- `--debug`: Enable debug mode with visualization output

### Training Features

**Lazy Loading:**
- Images are loaded on-demand from parquet files during training
- Enables training on very large datasets without loading everything into memory

**Index Caching:**
- Dataset indexing results are cached in `.index_cache/` directory
- Subsequent runs skip indexing and load from cache
- Cache is automatically invalidated if parquet files change

**Memory Efficiency:**
- Uses single-process data loading (`num_workers=0`) to prevent memory overflow
- Parquet file caching limited to 2 files at a time
- Efficient byte-to-image decoding

**Progress Logging:**
- Real-time training progress with tqdm progress bars
- Detailed logging with timestamps, loss, IoU, and learning rate
- Checkpoint saving every 5 epochs

### Example Training Output

```
2025-01-XX XX:XX:XX - INFO - Loading OpenFake dataset from: D:\dataset\OpenFake\data
2025-01-XX XX:XX:XX - INFO - Found 543 parquet files
2025-01-XX XX:XX:XX - INFO - Using 434 files for training (80%)
2025-01-XX XX:XX:XX - INFO - Loading cached index from: D:\dataset\OpenFake\.index_cache\train_index_xxxxx.pkl
2025-01-XX XX:XX:XX - INFO - Loaded 635138 samples from cache for train split
2025-01-XX XX:XX:XX - INFO - Training started for 100 epochs
2025-01-XX XX:XX:XX - INFO - Epoch 1/100 - Training started
Epoch 1/100: 100%|████████| 1984/1984 [XX:XX<XX:XX, X.XXit/s]
2025-01-XX XX:XX:XX - INFO - Epoch 1/100 - Train Loss: 0.5012 - Val Loss: 0.4890 - Val IoU: 0.6234 - LR: 0.003000
Checkpoint saved: output/openfake_dinov3_small_e5.pt
```

## Output Files

Training results are saved in the `output/` directory:
- `{exp_name}.pt`: Final model weights (LoRA + decoder)
- `{exp_name}_e{epoch}.pt`: Checkpoint every 5 epochs
- `{exp_name}_metrics.json`: Training metrics (train_loss, val_loss, val_iou)

## Troubleshooting

**Memory Error:**
- Reduce `--batch_size`
- Reduce `--img_dim`
- Ensure `num_workers=0` is used (automatic for openfake dataset)

**Slow Indexing:**
- First run indexes all parquet files (takes time)
- Subsequent runs use cached index (fast)
- Cache files are in `.index_cache/` directory

**Checkpoint Not Found:**
- Download DINOv3 weights from [official repository](https://github.com/facebookresearch/dinov3)
- Or use `--dinov3_repo_dir` to specify local repository path

## References

Siméoni, O., Vo, H. V., Seitzer, M., Baldassarre, F., Oquab, M., Jose, C., Khalidov, V., Szafraniec, M., Yi, S., Ramamonjisoa, M., Massa, F., Haziza, D., Wehrstedt, L., Wang, J., Darcet, T., Moutakanni, T., Sentana, L., Roberts, C., Vedaldi, A., … Bojanowski, P. (2025). DINOv3 (No. arXiv:2508.10104). arXiv. https://doi.org/10.48550/arXiv.2508.10104

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models (arXiv:2106.09685). arXiv. http://arxiv.org/abs/2106.09685
