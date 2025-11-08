import os
import zipfile
import logging
import urllib.request
import glob
import pandas as pd
import pickle
import hashlib
from typing import Optional

import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VOCSegmentation

from .corruption import get_corruption_transforms

VOC_COLORMAP = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
]


class PascalVOCDataset(VOCSegmentation):
    def __init__(
        self,
        root: str = "./data",
        year: str = "2012",
        image_set: str = "train",
        download: bool = True,
        transform: Optional[A.Compose] = None,
        use_index_label: bool = True,
    ):
        super().__init__(
            root=root,
            year=year,
            image_set=image_set,
            download=download,
            transform=transform,
        )
        self.n_classes = 21
        self.transform = transform
        self.use_index_label = use_index_label

    @staticmethod
    def _convert_to_segmentation_mask(
        mask: np.ndarray, use_index_label: bool = True
    ) -> np.ndarray:
        height, width = mask.shape[:2]
        segmentation_mask = np.zeros(
            (height, width, len(VOC_COLORMAP)),
            dtype=np.float32,
        )
        for label_index, label in enumerate(VOC_COLORMAP):
            segmentation_mask[:, :, label_index] = np.all(
                mask == label, axis=-1
            ).astype(float)

        if use_index_label:
            segmentation_mask = np.argmax(segmentation_mask, axis=-1)
        return segmentation_mask

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        image = cv2.imread(self.images[index])
        mask = cv2.imread(self.masks[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        mask = self._convert_to_segmentation_mask(mask, self.use_index_label)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        image = np.moveaxis(image, -1, 0) / 255
        return image, mask


class ADE20kDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "training",
        transform: Optional[A.Compose] = None,
    ):
        self.root = root
        self.split = split
        self.n_classes = 150
        self.transform = transform

        root = os.path.join(root, "ADEChallengeData2016")
        self.images_dir = os.path.join(root, "images", split)
        self.masks_dir = os.path.join(root, "annotations", split)

        # Check if the dataset is already downloaded
        if not os.path.exists(self.images_dir) or not os.path.exists(self.masks_dir):
            self.download_and_extract_dataset()

        self.image_files = os.listdir(self.images_dir)

    def download_and_extract_dataset(self) -> None:
        dataset_url = (
            "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
        )
        zip_path = os.path.join(self.root, "ADEChallengeData2016.zip")
        os.makedirs(self.root, exist_ok=True)

        logging.info("Downloading dataset...")
        urllib.request.urlretrieve(dataset_url, zip_path)

        logging.info("Extracting dataset...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(self.root)

        logging.info("Dataset extracted!")
        os.remove(zip_path)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        img_name = self.image_files[index]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name.replace(".jpg", ".png"))

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) - 1
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        image = np.moveaxis(image, -1, 0) / 255
        return image, mask


class OpenFakeDataset(Dataset):
    """OpenFake 데이터셋 로더. Parquet 파일에서 이미지를 lazy loading합니다."""
    
    @staticmethod
    def _bytes_to_image(byte_dict):
        """Parquet에 저장된 {'bytes': b'...'} 구조를 np.array 이미지로 변환"""
        try:
            if not isinstance(byte_dict, dict) or "bytes" not in byte_dict:
                return None
            img_bytes = byte_dict["bytes"]
            img_array = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        except Exception:
            return None
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[A.Compose] = None,
        n_classes: int = 2,
    ):
        self.root = root
        self.split = split
        self.n_classes = n_classes
        self.transform = transform
        
        # Parquet 파일 경로 찾기
        data_dir = os.path.join(root, "data")
        logging.info(f"Loading OpenFake dataset from: {data_dir}")
        
        parquet_files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
        logging.info(f"Found {len(parquet_files)} parquet files")
        
        if len(parquet_files) == 0:
            raise FileNotFoundError(f"No parquet files found in: {data_dir}")
        
        # Train/Validation split
        if split == "train":
            # 80% train
            split_idx = int(len(parquet_files) * 0.8)
            self.parquet_files = parquet_files[:split_idx]
            logging.info(f"Using {len(self.parquet_files)} files for training (80%)")
        else:  # val or test
            # 20% validation
            split_idx = int(len(parquet_files) * 0.8)
            self.parquet_files = parquet_files[split_idx:]
            logging.info(f"Using {len(self.parquet_files)} files for validation (20%)")
        
        # 인덱스 파일 경로 생성 (데이터셋 경로 기반 해시)
        index_cache_dir = os.path.join(root, ".index_cache")
        os.makedirs(index_cache_dir, exist_ok=True)
        
        # parquet 파일 목록의 해시를 생성하여 인덱스 파일명 결정
        parquet_hash = hashlib.md5("".join(sorted(self.parquet_files)).encode()).hexdigest()[:8]
        index_file = os.path.join(index_cache_dir, f"{split}_index_{parquet_hash}.pkl")
        
        # 저장된 인덱스가 있으면 로드
        if os.path.exists(index_file):
            logging.info(f"Loading cached index from: {index_file}")
            try:
                with open(index_file, "rb") as f:
                    cached_data = pickle.load(f)
                    # 캐시된 parquet 파일 목록과 현재 목록이 일치하는지 확인
                    if cached_data.get("parquet_files") == self.parquet_files:
                        self.data = cached_data["data"]
                        logging.info(f"Loaded {len(self.data)} samples from cache for {split} split")
                    else:
                        logging.info("Parquet files changed, re-indexing...")
                        self.data = None
                if self.data is not None:
                    self.parquet_cache = {}  # 최근 사용한 parquet 파일 캐시 (메모리 제한)
                    self.max_cache_size = 2  # 최대 캐시할 parquet 파일 수 (메모리 부족 방지)
                    return
            except Exception as e:
                logging.warning(f"Failed to load cached index: {e}, re-indexing...")
        
        # 인덱스가 없거나 변경되었으면 새로 생성
        logging.info(f"Indexing data from {len(self.parquet_files)} parquet files...")
        self.data = []
        self.parquet_cache = {}  # 최근 사용한 parquet 파일 캐시 (메모리 제한)
        self.max_cache_size = 2  # 최대 캐시할 parquet 파일 수 (메모리 부족 방지)
        
        for parquet_path in tqdm(self.parquet_files, desc=f"Indexing {split} parquet files"):
            try:
                # 필요한 컬럼만 읽기 (메모리 효율)
                df = pd.read_parquet(parquet_path, columns=["image", "prompt", "label"])
                
                # 각 row의 인덱스만 저장 (lazy loading)
                for idx, row in df.iterrows():
                    self.data.append({
                        "parquet_path": parquet_path,
                        "row_index": idx,
                        "label": row["label"],
                        "prompt": row["prompt"],  # 나중에 사용할 변수
                    })
            except Exception as e:
                logging.warning(f"Error loading {parquet_path}: {e}")
                continue
        
        logging.info(f"Total {len(self.data)} samples indexed for {split} split")
        
        # 인덱스 저장
        try:
            cache_data = {
                "parquet_files": self.parquet_files,
                "data": self.data,
            }
            with open(index_file, "wb") as f:
                pickle.dump(cache_data, f)
            logging.info(f"Index saved to: {index_file}")
        except Exception as e:
            logging.warning(f"Failed to save index cache: {e}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        sample = self.data[index]
        parquet_path = sample["parquet_path"]
        row_index = sample["row_index"]
        label = sample["label"]
        prompt = sample["prompt"]  # 나중에 사용할 변수 (일단 할당만)
        
        # Parquet 파일 읽기 (캐시에서 가져오거나 새로 읽기)
        # 멀티프로세싱 환경에서는 캐시가 공유되지 않으므로 주의
        # num_workers=0으로 설정하여 메인 프로세스에서만 읽도록 함
        if parquet_path in self.parquet_cache:
            df = self.parquet_cache[parquet_path]
        else:
            # 필요한 컬럼만 읽기 (메모리 효율)
            df = pd.read_parquet(parquet_path, columns=["image", "prompt", "label"])
            
            # 캐시 크기 제한 (LRU 방식) - 메모리 부족 방지를 위해 크게 줄임
            if len(self.parquet_cache) >= self.max_cache_size:
                # 가장 오래된 항목 제거
                oldest_key = next(iter(self.parquet_cache))
                del self.parquet_cache[oldest_key]
            self.parquet_cache[parquet_path] = df
        
        row = df.iloc[row_index]
        
        # 이미지 디코딩 (bytes -> numpy array)
        image = self._bytes_to_image(row["image"])
        if image is None:
            logging.warning(f"Failed to decode image at {parquet_path}, row {row_index}")
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Label을 기반으로 segmentation mask 생성
        # fake=1, real=0
        label_to_class = {"fake": 1, "real": 0}
        class_id = label_to_class.get(label.lower(), 1)  # 기본값은 1 (fake)
        
        # 전체 이미지를 해당 클래스로 설정
        mask = np.full((image.shape[0], image.shape[1]), class_id, dtype=np.int64)
        
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        
        image = np.moveaxis(image, -1, 0) / 255.0
        return image, mask


def get_dataloader(
    dataset_name: str,
    img_dim: tuple[int, int] = (490, 490),
    batch_size: int = 6,
    corruption_severity: int = None,
    dataset_root: str = None,
    n_classes: int = None,
) -> tuple[DataLoader, DataLoader]:
    """Get the dataloaders for Pascal VOC (voc), ADE20k (ade20k), or OpenFake (openfake)

    Args:
        dataset_name (str): The name of the dataset either, `voc`, `ade20k`, or `openfake`.
        img_dim (tuple[int, int], optional): The input size of the images.
            Defaults to (490, 490).
        batch_size (int, optional): The batch size of the dataloader. Defaults to 6.
        corruption_severity (int, optional): The corruption severity level between 1 and 5.
            Defaults to None.
        dataset_root (str, optional): Root directory for the dataset. Required for openfake.
        n_classes (int, optional): Number of classes. Required for openfake.

    Returns:
        tuple[DataLoader, DataLoader]: The train and validation loader respectively.
    """
    assert dataset_name in ["ade20k", "voc", "openfake"], "dataset name not in [ade20k, voc, openfake]"
    transform = A.Compose([A.Resize(height=img_dim[0], width=img_dim[1])])

    if dataset_name == "voc":
        train_dataset = PascalVOCDataset(
            root="./data",
            year="2012",
            image_set="train",
            download=False,
            transform=transform,
        )

        if corruption_severity is not None:
            transform = get_corruption_transforms(img_dim, corruption_severity)
        val_dataset = PascalVOCDataset(
            root="./data",
            year="2012",
            image_set="val",
            download=False,
            transform=transform,
        )
    elif dataset_name == "ade20k":
        train_dataset = ADE20kDataset(
            root="./data",
            split="training",
            transform=transform,
        )

        if corruption_severity is not None:
            transform = get_corruption_transforms(img_dim, corruption_severity)
        val_dataset = ADE20kDataset(
            root="./data",
            split="validation",
            transform=transform,
        )
    elif dataset_name == "openfake":
        if dataset_root is None:
            dataset_root = r"D:\dataset\OpenFake"
        if n_classes is None:
            n_classes = 2
        
        logging.info("=" * 60)
        logging.info("Loading OpenFake Training Dataset...")
        logging.info("=" * 60)
        train_dataset = OpenFakeDataset(
            root=dataset_root,
            split="train",
            transform=transform,
            n_classes=n_classes,
        )
        
        if corruption_severity is not None:
            transform = get_corruption_transforms(img_dim, corruption_severity)
        
        logging.info("=" * 60)
        logging.info("Loading OpenFake Validation Dataset...")
        logging.info("=" * 60)
        val_dataset = OpenFakeDataset(
            root=dataset_root,
            split="test",  # test_metadata.csv 사용
            transform=transform,
            n_classes=n_classes,
        )
        logging.info("=" * 60)
        logging.info(f"Dataset loading complete! Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        logging.info("=" * 60)

    # OpenFake 데이터셋은 메모리 사용량이 크므로 worker 수를 줄임
    num_workers = 0 if dataset_name == "openfake" else 32
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader
