import os
import requests
import zipfile
import shutil
from tqdm import tqdm
import tarfile

def download_file(url, filename):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def download_celeba():
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Download CelebA dataset (using the Align&Cropped Images version)
    print("Downloading CelebA dataset...")
    base_url = "https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html"
    img_url = "https://drive.google.com/uc?export=download&id=0B7EVK8r0v71pZjFTYXZWM3FlRnM"
    attr_url = "https://drive.google.com/uc?export=download&id=0B7EVK8r0v71pY0NsMzYyZDEzcDg"
    
    # Try alternative direct download links
    img_url = "https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADSNlYuUMhAZQXFRxQHqQkHa/Img/img_align_celeba.zip?dl=1"
    attr_url = "https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADSNlYuUMhAZQXFRxQHqQkHa/Anno/list_attr_celeba.txt?dl=1"
    
    # Download files
    img_file = 'data/img_align_celeba.zip'
    attr_file = 'data/list_attr_celeba.txt'
    
    print("Downloading images...")
    download_file(img_url, img_file)
    
    print("Downloading attributes...")
    download_file(attr_file, attr_file)
    
    # Extract the dataset
    print("Extracting dataset...")
    with zipfile.ZipFile(img_file, 'r') as zip_ref:
        zip_ref.extractall('data/celeba')
    
    # Create a proper directory structure for ImageFolder
    print("Organizing dataset...")
    os.makedirs('data/celeba/celeba', exist_ok=True)
    for img in tqdm(os.listdir('data/celeba/img_align_celeba')):
        shutil.move(
            os.path.join('data/celeba/img_align_celeba', img),
            os.path.join('data/celeba/celeba', img)
        )
    
    # Clean up
    shutil.rmtree('data/celeba/img_align_celeba')
    os.remove(img_file)
    
    print("Dataset preparation complete!")
    print("\nNote: The dataset is available for non-commercial research purposes only.")
    print("Please cite the paper: Deep Learning Face Attributes in the Wild (ICCV 2015)")

if __name__ == "__main__":
    download_celeba() 