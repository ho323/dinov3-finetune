import os
import io
import glob
import pandas as pd
from tqdm import tqdm
import numpy as np
import cv2
import torch
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# YOLOv11n 얼굴 탐지 모델 다운로드
model_path = hf_hub_download(repo_id="AdamCodd/YOLOv11n-face-detection", filename="model.pt")
model = YOLO(model_path).to("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 경로
data_dir = r"D:\dataset\OpenFake\data"
output_dir = r"D:\dataset\OpenFake\filtered"
os.makedirs(output_dir, exist_ok=True)

# 모든 parquet 파일
parquet_files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))

def bytes_to_image(byte_dict):
    """Parquet에 저장된 {'bytes': b'...'} 구조를 np.array 이미지로 변환"""
    try:
        if not isinstance(byte_dict, dict) or "bytes" not in byte_dict:
            return None
        img_bytes = byte_dict["bytes"]
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

# 3510개씩 이미지가 쌓이면 parquet로 저장
BATCH_SIZE = 3510
all_valid_rows = []
file_count = 0

for parquet_path in parquet_files:
    df = pd.read_parquet(parquet_path)

    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {os.path.basename(parquet_path)}"):
        img = bytes_to_image(row["image"])
        if img is None:
            continue

        try:
            results = model.predict(img, verbose=False)
            num_faces = len(results[0].boxes)
            if num_faces > 0:
                all_valid_rows.append(row)
        except Exception as e:
            print(f"Error on index {i}: {e}")

        # 3510개마다 저장
        if len(all_valid_rows) >= BATCH_SIZE:
            batch_df = pd.DataFrame(all_valid_rows[:BATCH_SIZE])
            out_name = f"filtered_batch_{file_count:05d}.parquet"
            batch_df.to_parquet(os.path.join(output_dir, out_name), index=False)
            print(f"저장: {out_name} ({BATCH_SIZE}개)")
            all_valid_rows = all_valid_rows[BATCH_SIZE:]
            file_count += 1

# 남은 데이터 저장
if len(all_valid_rows) > 0:
    batch_df = pd.DataFrame(all_valid_rows)
    out_name = f"filtered_batch_{file_count:05d}.parquet"
    batch_df.to_parquet(os.path.join(output_dir, out_name), index=False)
    print(f"저장: {out_name} ({len(batch_df)}개)")

print("✅ 얼굴이 1개 이상 탐지된 데이터셋(3510개 단위) 저장 완료!")
