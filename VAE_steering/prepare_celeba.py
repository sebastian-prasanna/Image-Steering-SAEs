import os
import zipfile
import shutil
from tqdm import tqdm

def prepare_celeba():
    """Prepare manually downloaded CelebA dataset"""
    # Check if files exist
    img_zip = 'data/img_align_celeba.zip'
    if not os.path.exists(img_zip):
        print(f"Error: {img_zip} not found!")
        print("Please download img_align_celeba.zip from https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")
        print("and place it in the 'data' folder")
        return
    
    # Create directories
    os.makedirs('data/celeba/celeba', exist_ok=True)
    
    # Extract the dataset
    print("Extracting dataset...")
    with zipfile.ZipFile(img_zip, 'r') as zip_ref:
        zip_ref.extractall('data/celeba')
    
    # Create a proper directory structure for ImageFolder
    print("Organizing dataset...")
    for img in tqdm(os.listdir('data/celeba/img_align_celeba')):
        shutil.move(
            os.path.join('data/celeba/img_align_celeba', img),
            os.path.join('data/celeba/celeba', img)
        )
    
    # Clean up
    shutil.rmtree('data/celeba/img_align_celeba')
    os.remove(img_zip)
    
    print("\nDataset preparation complete!")
    print("Note: The dataset is available for non-commercial research purposes only.")
    print("Please cite the paper: Deep Learning Face Attributes in the Wild (ICCV 2015)")

if __name__ == "__main__":
    prepare_celeba() 