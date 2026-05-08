# Diabetic Retinopathy Classification
This directory contains an ensemble of 5 DenseNet121 models for Diabetic Retinopathy grading.

## 1. Setup
Ensure you have a CUDA-enabled GPU, a Python environment, and install dependencies:
```
pip install -r requirements.txt
```

## 2. File structure
- Weights: the 5 models (best_model_fold[0-4].pth) are placed inside a /models folder. 
- README.md : a guide to run the inference.
- requirements.txt : a list of required python libraries.
- training.ipynb : Jupyter notebook used to train our models.
- infer.py : the inference script.
- model_config.json : configuration settings used to train the model.

## 3. Requirements

Add your input data:
- A CSV with image filenames and labels, the column names must be 'Image' and 'Label'
- A folder containing the raw images

## 4. Running Inference
Run the script by providing the paths to your data:

```
python infer.py --csv_path test_labels.csv --images_path images_dir --output_filename submission.csv
```
Outputs:
- submission.csv: Image file name, Actual labels vs Predictions;
- confusion_matrix.png: Visual performance summary;
