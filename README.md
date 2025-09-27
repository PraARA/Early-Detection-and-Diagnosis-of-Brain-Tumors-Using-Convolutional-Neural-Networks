# Brain Tumor Detection using CNN (TensorFlow / Keras)

This project provides a ready-to-run template to build a **brain tumor detection** system using Convolutional Neural Networks (CNNs) with TensorFlow/Keras.

**What’s included**
- `train.py` — full training pipeline (patient-level split, preprocessing, model training, fine-tuning, save model)
- `predict.py` — simple inference script to load model and predict a single image
- `utils.py` — helper functions (data mapping, tf.data pipeline, Grad-CAM)
- `requirements.txt` — Python dependencies
- `sample_run.sh` — quick commands to run training and inference
- `README.md` — this file

> **Note:** This repository does NOT include MRI images. Download a dataset (Kaggle link below) and extract into `dataset/` following the structure described.

## Recommended dataset (example)
Kaggle — "Brain MRI Images for Brain Tumor Detection":  
https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri

After downloading, unzip into this project root as:
```
brain_tumor_cnn_project/
  dataset/
    yes/    # tumor images
    no/     # no-tumor images
```

## Quick setup (Linux / Mac / WSL / VS Code)
```bash
python -m venv venv
source venv/bin/activate       # or venv\Scripts\activate on Windows
pip install -r requirements.txt
# put dataset in ./dataset as described
python train.py                # training (will create best_model.h5)
python predict.py --image path/to/test.jpg  # run inference
```

## Notes
- The training script uses **EfficientNetB0** transfer learning (ImageNet weights).
- Splits are done at **patient level** if patient IDs can be parsed from filenames. Otherwise image-level split is used — better to adjust filenames to include patient IDs to avoid leakage.
- For large datasets, use a GPU (Colab / local CUDA) for faster training.
- See `utils.py` for Grad-CAM visualization helper.

