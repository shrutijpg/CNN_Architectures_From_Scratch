# 🧠 CIFAR-10 Image Classification using CNN Architectures

This project implements and compares four classic Convolutional Neural Network (CNN) architectures on the **CIFAR-10 dataset**. It provides a side-by-side evaluation of their performance in terms of accuracy and loss.

---

## 📌 Models Implemented

- ✅ **LeNet**
- ✅ **AlexNet**
- ✅ **VGG (Mini Version)**
- ✅ **ResNet (Shallow Custom Version)**

---

## 📂 Project Structure
├── app.py # Main training script for all models  
├── alexnet.py # AlexNet architecture  
├── LeNet.py # LeNet architecture  
├── VGG.py # VGG architecture (mini version)  
├── resnet.py # Custom ResNet with residual blocks  
├── save_models/ # Directory to save best .h5 models  
├── results/ # Accuracy/loss plots and bar charts  
│ ├── AlexNet_metrics.png  
│ ├── LeNet_metrics.png  
│ ├── VGG_metrics.png    
│ ├── ResNet_metrics.png  
│ └── accuracy_comparison.png  


---

## 📊 Dataset Info

- **Dataset**: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
- **Size**: 60,000 images (50k train + 10k test)
- **Classes**: 10 (aeroplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Image Size**: 32x32 RGB

---

## 🚀 How to Run

Make sure you have Python 3.7+ and TensorFlow installed:


## 📈 Output Results

### ✅ Accuracy Comparison (Bar Chart)
- Automatically saved in `results/accuracy_comparison.png`

### 📉 Individual Training Metrics (Saved per model)
- Training vs Validation Accuracy and Loss
- Saved in `results/{ModelName}_metrics.png`

---

## 📊 Test Accuracy Summary

| Model   | Test Accuracy |
|---------|----------------|
| AlexNet | ~71%           |
| LeNet   | ~45%           |
| VGG     | ~10%          |
| ResNet  | ~80%          |

> ℹ️ **Note**: VGG is underfitting — likely due to over-parameterization without enough regularization for CIFAR-10.  
> LeNet is trained on grayscale inputs and performs reasonably for its simplicity.

---

## 🛠️ Features

- ✅ `ModelCheckpoint` to save best weights
- ✅ Train/Validation split (80/20)
- ✅ Accuracy & loss plots for each model
- ✅ Final model comparison bar chart

---

## 🧪 Training Configurations

- **Optimizer**: Adam  
- **Loss Function**: Sparse Categorical Crossentropy  
- **Epochs**: 20 (can be changed)  
- **Batch Size**: 64  
- **Validation Split**: 20%

---

## ✅ Future Improvements

- [ ] Add Batch Normalization to all models  
- [ ] Use Data Augmentation (e.g., rotation, flipping)  
- [ ] Tune VGG architecture to suit CIFAR-10 better  
- [ ] Add learning rate scheduling  
- [ ] Try pretrained models (like ResNet50 via transfer learning)

---

## 🧑‍💻 Author

**Shruti Bhandarkar**  
📧 *[https://www.linkedin.com/in/shruti-b-51a3a9263/]*

---

## 📝 License

This project is open source and available under the **MIT License**.

---

## 🙌 Acknowledgements

- TensorFlow and Keras for deep learning APIs  
- CIFAR-10 dataset from the University of Toronto  
- Architecture references from original research papers







