# 🧠 ASL Alphabet Detection with CNN (TensorFlow/Keras)

> A deep learning project to classify American Sign Language (ASL) alphabets using a Convolutional Neural Network (CNN) trained on the [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet).

---

## 📂 Dataset

We used the publicly available ASL dataset from Kaggle:

🔗 **Download Link:** [ASL Alphabet Dataset – Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

**Structure:**

```
data/
├── asl_alphabet_train/
│   └── asl_alphabet_train/
│       ├── A/
│       ├── B/
│       └── ... Z/
└── asl_alphabet_test/
    └── asl_alphabet_test/
        ├── A_test.jpg
        ├── B_test.jpg
        └── ... Z_test.jpg
```

---

## 🛠️ Features

- CNN model trained using Keras and TensorFlow
- Data augmentation using `ImageDataGenerator`
- Classification report & accuracy/loss graphs
- Inference on test images
- Easily extendable to real-time webcam detection

---

## 🚀 How to Run

1. **Clone this repository:**
```bash
git clone https://github.com/Devikareddykubdavaram/asl_detection.git
cd asl-detection
```

2. **Download the dataset** from Kaggle and place it in:
```
/data/asl_alphabet_train/asl_alphabet_train/
```
and test images in:
```
/data/asl_alphabet_test/asl_alphabet_test/
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run the notebook:**
Launch the Jupyter notebook and execute all cells:
```bash
jupyter notebook ASL_Detection.ipynb
```

---

## 🧠 Model Architecture

- **2 Convolution Layers** with ReLU + MaxPooling
- **Flatten → Dense Layer (128 units)**
- **Dropout (0.5)**
- **Output Layer** with softmax (29 classes)

---

## 📈 Sample Output

| Test Image | Prediction |
|------------|------------|
| A.jpg      | A          |
| C.jpg      | C          |
| ...        | ...        |

Also includes graphs for:
- Training vs Validation Accuracy
- Training vs Validation Loss

---

## 📦 Future Enhancements

- [ ] Real-time webcam detection using OpenCV
- [ ] Deploy as a web app using Streamlit or Flask
- [ ] Add support for full ASL words

---

## 🤝 Acknowledgements

- [Kaggle: ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- TensorFlow & Keras for the deep learning framework
- OpenCV (planned for real-time extension)

---
📌 This project was developed as part of my internship at **Unified Mentor** under the guidance of industry experts.

