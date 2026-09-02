"""
Script tải dữ liệu Oxford-IIIT Pet Dataset và tự động phân loại Chó/Mèo.
Nguồn: https://www.robots.ox.ac.uk/~vgg/data/pets/
Chúng ta sẽ lấy một tập con nhỏ (500 Mèo, 500 Chó) để demo pipeline chạy nhanh.
"""

import os
import tarfile
import shutil
import requests
from pathlib import Path

DATA_URL = "https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz"
TAR_PATH = "pets.tar.gz"
RAW_DIR = Path("data/raw")

def download_and_extract():
    print("Đang tải dữ liệu Oxford-IIIT Pets (khoảng 790MB, có thể mất 1-2 phút)...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(DATA_URL, headers=headers, stream=True)
    response.raise_for_status()
    
    with open(TAR_PATH, 'wb') as f:
        # Ghi file tải về
        for chunk in response.iter_content(chunk_size=8192*4):
            f.write(chunk)
            
    print("Đang giải nén...")
    with tarfile.open(TAR_PATH, "r:gz") as tar:
        tar.extractall("temp_data")
        
    print("Đang sắp xếp thư mục (chỉ lấy 500 ảnh mỗi loại để train nhanh)...")
    (RAW_DIR / "Cat").mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "Dog").mkdir(parents=True, exist_ok=True)
    
    cat_images = []
    dog_images = []
    
    # Dataset này quy ước: Tên loài Mèo bắt đầu bằng chữ IN HOA, Chó bằng chữ in thường
    for img_file in Path("temp_data/images").glob("*.jpg"):
        if img_file.name[0].isupper():
            cat_images.append(img_file)
        else:
            dog_images.append(img_file)
            
    # Lấy 500 ảnh mỗi loại để tiết kiệm thời gian train
    for img in cat_images[:500]:
        shutil.copy(str(img), str(RAW_DIR / "Cat" / img.name))
    for img in dog_images[:500]:
        shutil.copy(str(img), str(RAW_DIR / "Dog" / img.name))
        
    # Dọn dẹp file rác
    os.remove(TAR_PATH)
    shutil.rmtree("temp_data")
    
    cat_count = len(list((RAW_DIR / "Cat").glob("*.jpg")))
    dog_count = len(list((RAW_DIR / "Dog").glob("*.jpg")))
    
    print("✅ Hoàn tất!")
    print(f"Đã chuẩn bị xong: {cat_count} ảnh Mèo, {dog_count} ảnh Chó tại thư mục {RAW_DIR}")

if __name__ == "__main__":
    download_and_extract()
