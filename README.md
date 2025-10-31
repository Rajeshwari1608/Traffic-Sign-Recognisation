# 🚦 Simple Traffic Sign Recognition (Using Preprocessed Pickle Dataset)

This beginner-friendly deep learning project recognizes traffic signs using the preprocessed Kaggle dataset:
https://www.kaggle.com/datasets/valentynsichkar/traffic-signs-preprocessed

## 📂 Dataset Files
- train.pickle
- valid.pickle
- test.pickle

## 🧠 Model
A small CNN with:
- 2 Conv2D layers
- 2 MaxPooling layers
- 1 Dense hidden layer
- Softmax output for 43 classes

## 🚀 How to Run
1. Download `.pickle` files and place them under `dataset/`
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
