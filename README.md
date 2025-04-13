# STM-property-prediction
Molecular property prediction based on STM imaging

## Project Overview
This project provides a deep learning framework for image processing and molecular structure reconstruction. It includes:

1. Pix2Pix Model: A Generative Adversarial Network (GAN) for image-to-image translation.
2. Molecular Structure Reconstruction: Rebuild molecular structures from density maps.
3. DimeNet Model: A graph neural network for predicting molecular properties (e.g., polarizability).

## Installation
Ensure the following dependencies are installed:

pip install torch torchvision numpy opencv-python scikit-learn torch-geometric

## Data Preparation

### 1. Pix2Pix Data
- Store input and label images in the ./datasets directory.
- Preprocess data using the save_dataset_as_vectors function in utils.py.

### 2. DimeNet Data
- Store molecular data in XYZ format and properties in CSV files.
- Load data using the load_data function in dimnet_util.py.

## Usage

### 1. Vectorizing Data with `utils.py`
Before training the Pix2Pix model, you need to vectorize your data using the `utils.py` script. This script converts raw images and XYZ files into a vector format suitable for training.

#### Steps to Vectorize Data:
1. **Organize Your Data**:
   - Place your input images and corresponding XYZ files in a structured directory. For example:
     ```
     ./data/
     ├── stm_image_1148/
     │   ├── image1.jpg
     │   ├── image2.jpg
     │   └── data.xyz
     └── stm_image_908/
         ├── image1.jpg
         ├── image2.jpg
         └── data.xyz
     ```

2. **Run the Vectorization Script**:
   ```bash
   python utils.py
   ```
### 2. Training the Pix2Pix Model
```bash
python multigpu_pix2pix.py
```
- The script loads preprocessed data from ./datasets/vector.
- Trained models are saved to ./checkpoints/saved_pix2pix.pt.

### 3. Testing the Pix2Pix Model
```bash
python test_pix2pix.py --model ./checkpoints/saved_pix2pix.pt --file_num n
```
- Input: fign_resize.jpg
- Output: fign_output.jpg

### 4. Rebuilding Molecular Structures
```bash
python locate.py n
```
- Input: fign_output.jpg
- Output: rebuild_fign.png

### 5. Training the DimeNet Model
```bash
python dimenet_run.py
```
- Loads data from ../stm_image_908 and ../stm_image_1148.
- Best model is saved as best_dimenet.pt.
