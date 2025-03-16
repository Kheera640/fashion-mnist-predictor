# Fashion MNIST Classifier

A TensorFlow-based neural network for classifying Fashion MNIST images. The project includes data preprocessing, model training, evaluation, and visualization tools. A pre-trained model (`fashionMNIST.h5`) is also available for quick inference.

## Features
- **Data Preprocessing:** Normalization using `MinMaxScaler`.
- **Model Training:** A simple dense neural network.
- **Evaluation:** Accuracy, confusion matrix.
- **Visualization:** Random image prediction.
- **Pre-trained Model:** Use without retraining.

## Installation
Clone the repository:
```sh
git clone https://github.com/yourusername/fashion-mnist-classifier.git
cd fashion-mnist-classifier
```

Install dependencies:
```sh
pip install -r requirements.txt
```

## Usage
### 1. Running the Model Training (Optional)
If you want to train the model yourself, run:
```sh
python main.py
```
This will train the model and save it as `fashionMNIST.h5`.

### 2. Using the Pre-trained Model for Prediction
To test the model on a random image, run:
```sh
python plot_helper.py
```

## File Structure
```
.
├── main.py          # Preprocesses data, trains & evaluates model
├── plot_helper.py   # Loads saved model & plots random image prediction
├── fashionMNIST.h5  # Pre-trained model
├── requirements.txt # Required dependencies
└── README.md        # Instructions
```

## Dependencies
- TensorFlow
- scikit-learn
- NumPy
- Matplotlib

Install them using:
```sh
pip install tensorflow scikit-learn numpy matplotlib
```

## License
This project is licensed under the MIT License.

